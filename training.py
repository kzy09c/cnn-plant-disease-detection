"""Two standalone binary recipes; saved test results are never overwritten."""
import argparse
import csv
import math

import numpy as np
import torch
from torch import nn

from datasets import load_dataset
from io_utils import ROOT, read_csv, save, sha256
from model import CHANNELS, CLASSES, TomatoCNN, prepare_batch, seed_everything
from metrics import evaluate, bootstrap, predict


def balanced_batches(pools, batch_size, rng):
    half = batch_size // 2
    draws = math.ceil(max(map(len, pools)) / half) * half
    sampled = [np.concatenate([rng.permutation(pool) for _ in range(math.ceil(draws / len(pool)))])[:draws] for pool in pools]
    return [rng.permutation(np.concatenate([p[start:start+half] for p in sampled])) for start in range(0, draws, half)]


def fit(mode, pretrained, images, targets, labels, training, validation, args, output, name, epochs, seed):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = TomatoCNN().cuda().to(memory_format=torch.channels_last)
    if mode == "plantvillage_features":
        shared = {k:v for k,v in pretrained["state_dict"].items() if not k.startswith("classifier.5.")}
        missing = model.load_state_dict(shared, strict=False)
        assert set(missing.missing_keys) == {"classifier.5.weight", "classifier.5.bias"} and not missing.unexpected_keys
        optimizer = torch.optim.AdamW([{"params":[p for n,p in model.named_parameters() if not n.startswith("classifier.5.")],"lr":.0001},
                                       {"params":model.classifier[5].parameters(),"lr":.001}], weight_decay=.0001)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    counts = np.bincount(labels[training], minlength=2)
    weights = torch.tensor(len(training)/(2*counts), device="cuda", dtype=torch.float32) if args.dataset == "plantvillage" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.000001)
    scaler = torch.amp.GradScaler("cuda")
    pools = [training[labels[training] == c] for c in (0,1)]
    history = []
    with (output / f"{name}_draws.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["epoch","batch","index","binary_class"])
        for epoch in range(1, epochs+1):
            model.train()
            if args.dataset == "bangladesh":
                batches = balanced_batches(pools, args.batch_size, rng)
            else:
                shuffled = torch.as_tensor(training, device="cuda")[torch.randperm(len(training), device="cuda")].cpu().numpy()
                batches = [shuffled[start:start+args.batch_size] for start in range(0,len(training),args.batch_size)]
            total = mass_total = correct = seen = 0
            optimizer_steps = 0
            seen_counts = np.zeros(2,dtype=int)
            for number, ids in enumerate(batches,1):
                if args.dataset == "bangladesh":
                    assert np.bincount(labels[ids],minlength=2).tolist() == [args.batch_size//2]*2
                for i in ids:
                    writer.writerow([epoch,number,i,CLASSES[labels[i]]])
                batch = torch.as_tensor(ids,device="cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits = model(prepare_batch(images,batch,True))
                    loss = criterion(logits, targets[batch])
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                optimizer_steps += int(scaler.get_scale() >= previous_scale)
                mass = weights[targets[batch]].sum().item() if weights is not None else len(ids)
                total += loss.item()*mass; mass_total += mass
                correct += (logits.argmax(1)==targets[batch]).sum().item(); seen += len(ids)
                seen_counts += np.bincount(labels[ids],minlength=2)
            # AMP can skip an update while reducing an overflowing loss scale.
            if optimizer_steps:
                schedule.step()
            row = {"epoch":epoch,"train_loss":total/mass_total,"train_augmented_accuracy":correct/seen,
                   "healthy_draws":int(seen_counts[0]),"unhealthy_draws":int(seen_counts[1]),"optimizer_steps":optimizer_steps}
            if validation is not None:
                probabilities = predict(model,images,validation,args.batch_size)
                measured = evaluate(labels[validation], probabilities)
                row.update({key:measured[key] for key in ("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy")})
            history.append(row)
            save(output / f"{name}_history.json",history)
            print(name,epoch,round(row["train_loss"],4),round(row.get("balanced_accuracy",0),4),flush=True)
    return model,history


def write_predictions(path, rows, ids, labels, probabilities):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path","group","true_class","predicted_class","probability_Healthy","probability_Unhealthy"])
        for i,scores in zip(ids,probabilities):
            writer.writerow([rows[i]["path"],rows[i]["group"],CLASSES[labels[i]],CLASSES[scores.argmax()],*scores.tolist()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["plantvillage","bangladesh"],default="plantvillage")
    parser.add_argument("--output",type=str,help="Fresh results directory, e.g. runs/plantvillage")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--cv-epochs",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--image-size",type=int,default=128)
    parser.add_argument("--audit-only",action="store_true")
    args = parser.parse_args()
    output = ROOT / (args.output or ("outputs" if args.dataset=="plantvillage" else "bangladesh_outputs"))
    if (output / "test_metrics.json").exists():
        print("Completed results preserved. Use --output runs/NAME for a fresh experiment.")
        return
    assert args.batch_size >= 2 and args.batch_size%2==0 and min(args.epochs,args.cv_epochs)>0
    output.mkdir(parents=True,exist_ok=True)
    if (output / "experiment.json").exists():
        raise FileExistsError("An interrupted experiment exists here; use a new output directory to preserve it")
    rows,arrays,labels,groups,development,test,released,folds,audit = load_dataset(args.dataset,args.image_size,output)
    if args.audit_only:
        print(audit); return
    assert torch.cuda.is_available(),"Training requires a CUDA GPU; prediction and saved-result review work on CPU"
    torch.set_num_threads(6); torch.backends.cudnn.benchmark=True
    pretrained = torch.load(ROOT/"outputs/tomato_cnn.pt",map_location="cpu",weights_only=True) if args.dataset=="bangladesh" else None
    record = {"arguments":vars(args),"classes":CLASSES,"channels":CHANNELS,"seed":42,
              "selection":"Fixed scratch recipe" if args.dataset=="plantvillage" else "Highest mean final-epoch grouped-CV balanced accuracy, then macro F1",
              "pretrained_sha256":sha256(ROOT/"outputs/tomato_cnn.pt") if pretrained is not None else None,
              "manifest_sha256":sha256(output/"split_manifest.csv"),"test":"Fixed argmax; no test-based selection"}
    save(output/"experiment.json",record)
    images = torch.from_numpy(arrays).cuda(); del arrays
    targets = torch.as_tensor(labels,device="cuda")
    modes = ["scratch"] if args.dataset=="plantvillage" else ["scratch","plantvillage_features"]
    cv = {}
    for mode in modes:
        final_epochs = []
        for fold,(training,validation) in enumerate(folds,1):
            model,history = fit(mode,pretrained,images,targets,labels,training,validation,args,output,f"{mode}_fold{fold}",args.cv_epochs,42+fold)
            final_epochs.append(history[-1]); del model
        cv[mode] = {"folds":final_epochs,**{"mean_"+key:float(np.mean([r[key] for r in final_epochs])) for key in ("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy")}}
        save(output/"cross_validation.json",cv)
    selected = max(modes,key=lambda m:(cv[m]["mean_balanced_accuracy"],cv[m]["mean_macro_f1"]))
    model,_ = fit(selected,pretrained,images,targets,labels,development,None,args,output,"final",args.epochs,142)
    filename = "tomato_cnn.pt" if args.dataset=="plantvillage" else "tomato_cnn_bangladesh.pt"
    model_path = output/filename
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"channels":CHANNELS,"classes":CLASSES,
                "image_size":args.image_size,"initialization":selected,"dataset":args.dataset,
                "scope":"Single tomato leaf health; independent performance on new gardens not established"},model_path)
    probabilities = predict(model,images,released,args.batch_size)
    positions = {i:j for j,i in enumerate(released)}
    primary = probabilities[[positions[i] for i in test]]
    result = evaluate(labels[test],primary)
    result.update(test_images=len(test),checkpoint_sha256=sha256(model_path),selected_initialization=selected,
                  confidence_intervals=bootstrap(labels[test],primary.argmax(1),groups[test]),
                  always_unhealthy_accuracy=float(np.mean(labels[test]==1)),parameter_count=sum(p.numel() for p in model.parameters()))
    write_predictions(output/"test_predictions.csv",rows,test,labels,primary)
    if args.dataset=="bangladesh":
        secondary = evaluate(labels[released],probabilities)
        secondary.update(test_images=len(released),warning="Full released holdout filename labels include annotation conflicts; descriptive only")
        save(output/"released_holdout_metrics.json",secondary)
        write_predictions(output/"released_holdout_predictions.csv",rows,released,labels,probabilities)
    save(output/"test_metrics.json",result)
    print(result)


if __name__ == "__main__":
    main()
