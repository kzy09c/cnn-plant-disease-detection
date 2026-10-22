"""Fine-tune the complete PlantVillage binary CNN on Bangladesh development data."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from io_utils import ROOT, read_csv, save, sha256
from metrics import bootstrap, evaluate, predict
from model import CHANNELS, CLASSES
from training import fit, write_predictions
from transfer_config import RECIPES

DEFAULT_OUTPUT = ROOT / "bangladesh_finetuned_outputs"
MODEL_FILENAME = "tomato_cnn_finetuned.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained", type=Path, default=ROOT/"outputs/tomato_cnn.pt")
    parser.add_argument("--cv-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.dataset, args.image_size = "bangladesh", 128
    if args.batch_size < 2 or args.batch_size % 2 or min(args.cv_epochs,args.epochs) < 1:
        parser.error("Use an even positive batch size and positive epoch counts")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; use a new --output directory to preserve results")
    if not torch.cuda.is_available():
        parser.error("Fresh training requires a CUDA GPU; saved-model prediction works on CPU")
    pretrained = torch.load(args.pretrained,map_location="cpu",weights_only=True)
    if sha256(args.pretrained) != sha256(ROOT/"outputs/tomato_cnn.pt"):
        parser.error("Use the recorded PlantVillage checkpoint; a different pretraining run needs a new provenance/overlap audit")
    assert pretrained["dataset"] == "plantvillage" and pretrained["classes"] == CLASSES
    output.mkdir(parents=True,exist_ok=True)
    rows, arrays, labels, groups, development, test, released, folds, audit = load_dataset("bangladesh",128,output)
    original = read_csv(ROOT/"bangladesh_outputs/split_manifest.csv")
    fields = ("path","group","split","binary_class","cv_validation_fold","exclusion","sha256","pixel_sha256")
    assert [{k:str(r.get(k,"")) for k in fields} for r in rows] == [{k:str(r.get(k,"")) for k in fields} for r in original], "Bangladesh cohort/folds changed"
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("output","pretrained")},
        "classes":CLASSES,"channels":CHANNELS,"seed":42,"gpu":torch.cuda.get_device_name(0),
        "stages":["Existing PlantVillage binary model trained on 14528 development images; 3632 test images excluded from training",
                  "Initialize every fold and final fit independently from that exact full checkpoint; retain all layers including the binary head",
                  "Fine-tune all parameters and BatchNorm statistics using only eligible Bangladesh development images"],
        "pretrained_checkpoint":"outputs/tomato_cnn.pt","pretrained_sha256":sha256(args.pretrained),
        "plantvillage_manifest_sha256":sha256(ROOT/"outputs/split_manifest.csv"),
        "plantvillage_experiment_sha256":sha256(ROOT/"outputs/experiment.json"),
        "manifest_sha256":sha256(output/"split_manifest.csv"),"recipes":RECIPES,
        "selection":"Highest mean final-epoch balanced accuracy over five grouped Bangladesh development folds; macro F1 breaks a tie",
        "sampling":"Exactly 32 Healthy/32 Unhealthy per default 64-image batch; cycle pools; unweighted cross entropy",
        "augmentation":"Original train-only flips/transposition, brightness and colour jitter; original normalization [-1,1]",
        "optimizer":"AdamW, weight decay 0.0001, cosine schedule to 0.000001, CUDA mixed precision",
        "test_policy":"Fixed argmax. Test inference begins only after recipe selection and final checkpoint saving. The same previously inspected Bangladesh holdout is reused: exploratory comparison, not a new independent test.",
        "budget_note":"20 CV epochs and 30 final epochs by default; the archived scratch comparison used 10/20, so this does not isolate initialization at equal budget.",
    }
    save(output/"experiment.json",record)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=True
    images=torch.from_numpy(arrays).cuda();del arrays
    targets=torch.as_tensor(labels,device="cuda")
    cv={}
    for recipe in RECIPES:
        last=[]
        for fold,(training,validation) in enumerate(folds,1):
            model,history=fit(recipe,pretrained,images,targets,labels,training,validation,args,output,
                              f"{recipe}_fold{fold}",args.cv_epochs,42+fold)
            last.append(history[-1]);del model
        cv[recipe]={"folds":last}
        for key in ("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy"):
            values=[r[key] for r in last]
            cv[recipe]["mean_"+key]=float(np.mean(values))
            cv[recipe]["sd_"+key]=float(np.std(values))
        save(output/"cross_validation.json",cv)
    selected=max(cv,key=lambda k:(cv[k]["mean_balanced_accuracy"],cv[k]["mean_macro_f1"]))
    # Commit the selection to disk before training/evaluating the final model.
    save(output/"selection.json",{"selected_initialization":selected,"selected_utc":datetime.now(timezone.utc).isoformat(),
                                 "criterion":record["selection"],"test_predictions_available":False})
    model,_=fit(selected,pretrained,images,targets,labels,development,None,args,output,"final",args.epochs,142)
    model_path=output/MODEL_FILENAME
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"channels":CHANNELS,"classes":CLASSES,
                "image_size":128,"normalization":{"mean":[.5]*3,"std":[.5]*3},
                "initialization":selected,"dataset":"plantvillage_then_bangladesh",
                "pretrained_sha256":record["pretrained_sha256"],"binary_head_retained":True,
                "scope":"Tomato leaf Healthy/Unhealthy after PlantVillage pretraining and Bangladesh fine-tuning; new-garden performance unmeasured"},model_path)
    # One forward pass over the released holdout supplies both reporting cohorts.
    probabilities=predict(model,images,released,args.batch_size)
    positions={i:j for j,i in enumerate(released)}
    primary=probabilities[[positions[i] for i in test]]
    result=evaluate(labels[test],primary)
    result.update(test_images=len(test),checkpoint_sha256=sha256(model_path),selected_initialization=selected,
                  confidence_intervals=bootstrap(labels[test],primary.argmax(1),groups[test]),
                  always_unhealthy_accuracy=float(np.mean(labels[test]==1)),parameter_count=sum(p.numel() for p in model.parameters()),
                  test_status="Reused, previously inspected holdout; exploratory comparison")
    write_predictions(output/"test_predictions.csv",rows,test,labels,primary)
    full=evaluate(labels[released],probabilities)
    full.update(test_images=len(released),warning="21 filename labels conflict with annotations; descriptive only; reused holdout")
    write_predictions(output/"released_holdout_predictions.csv",rows,released,labels,probabilities)
    save(output/"released_holdout_metrics.json",full)
    save(output/"confidence_intervals.json",result["confidence_intervals"])
    save(output/"test_metrics.json",result)
    save(output/"completion.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"checkpoint_sha256":sha256(model_path)})
    print(result,flush=True)


if __name__=="__main__":
    main()
