"""Validation-only recipe search for a custom CNN trained from random weights."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from model import LeafCNN
from prepare_data import ROOT, CONFIG, read_csv, write_csv, save_json

RECIPES={
    "mixed_mild":{"strong":False,"attention":False,"mixup":0.,"lr":.001},
    "mixed_strong":{"strong":True,"attention":False,"mixup":.15,"lr":.001},
    "mixed_attention":{"strong":True,"attention":True,"mixup":.10,"lr":.0007},
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def metrics(y,p):
    pred=np.asarray(p)>=.5
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {"n":len(y),"accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "healthy_recall":float(cm[0,0]/max(1,cm[0].sum())),
            "unhealthy_recall":float(cm[1,1]/max(1,cm[1].sum())),
            "confusion_matrix":cm.tolist()}

def selection_score(results):
    scores=[x["balanced_accuracy"] for x in results.values()]
    return .6*float(np.mean(scores))+.4*min(scores)

def augment(x,strong):
    x=x.float()/255
    n=len(x)
    for dim in [2,3]:
        flip=torch.rand(n,device=x.device)<.5
        x[flip]=x[flip].flip(dim)
    if strong:
        # Random affine crop/rotation in one interpolation operation.
        angle=(torch.rand(n,device=x.device)-.5)*math.pi/3
        scale=torch.empty(n,device=x.device).uniform_(.65,1.)
        theta=torch.zeros(n,2,3,device=x.device)
        theta[:,0,0]=angle.cos()*scale
        theta[:,0,1]=-angle.sin()*scale
        theta[:,1,0]=angle.sin()*scale
        theta[:,1,1]=angle.cos()*scale
        theta[:,:,2]=torch.empty(n,2,device=x.device).uniform_(-.12,.12)
        grid=F.affine_grid(theta,x.shape,align_corners=False)
        x=F.grid_sample(x,grid,padding_mode="reflection",align_corners=False)
        gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
        saturation=torch.empty(n,1,1,1,device=x.device).uniform_(.5,1.5)
        x=gray+(x-gray)*saturation
        mean=x.mean((2,3),keepdim=True)
        contrast=torch.empty(n,1,1,1,device=x.device).uniform_(.7,1.3)
        x=(x-mean)*contrast+mean
        gray_mask=torch.rand(n,device=x.device)<.07
        x[gray_mask]=gray[gray_mask].expand(-1,3,-1,-1)
        blur=torch.rand(n,device=x.device)<.1
        x[blur]=F.avg_pool2d(x[blur],3,stride=1,padding=1)
    strength=.25 if strong else .15
    brightness=torch.empty(n,1,1,1,device=x.device).uniform_(1-strength,1+strength)
    color=torch.empty(n,3,1,1,device=x.device).uniform_(.9 if strong else .95,1.1 if strong else 1.05)
    x=(x*brightness*color).clamp_(0,1)
    if strong and random.random()<.3:
        h=random.randint(8,32)
        w=random.randint(8,32)
        top=random.randint(0,x.shape[2]-h)
        left=random.randint(0,x.shape[3]-w)
        mask=torch.rand(n,device=x.device)<.3
        x[mask,:,top:top+h,left:left+w]=.5
    return ((x-.5)/.5).contiguous(memory_format=torch.channels_last)

@torch.inference_mode()
def probabilities(model,images,indices,batch=128):
    model.eval()
    output=[]
    for start in range(0,len(indices),batch):
        x=images[indices[start:start+batch]].float()/127.5-1
        with torch.autocast("cuda",dtype=torch.float16):
            p=model(x.contiguous(memory_format=torch.channels_last)).float().softmax(1)[:,1]
        output.extend(p.cpu().tolist())
    return np.asarray(output)

def train_one(name,recipe,images,rows,args):
    out=ROOT/"runs"/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        print("Using completed run",name,flush=True)
        return json.loads((out/"complete.json").read_text())
    assert not (out/"best.pt").exists(), "Use a new run name instead of overwriting an unfinished experiment"
    seed_all(args.seed)
    labels=torch.tensor([int(r["label"]) for r in rows],device="cuda")
    sources=sorted({r["source"] for r in rows if r["split"]=="train"})
    buckets=[torch.tensor([i for i,r in enumerate(rows) if r["source"]==s and r["split"]=="train" and int(r["label"])==c],device="cuda") for s in sources for c in [0,1]]
    assert all(len(b)>0 for b in buckets)
    assert args.batch_size%len(buckets)==0
    validation={s:np.array([i for i,r in enumerate(rows) if r["source"]==s and r["split"]=="validation"]) for s in sources}
    model=LeafCNN(attention=recipe["attention"]).cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["lr"],weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda")
    configuration={"recipe":recipe,"seed":args.seed,"epochs":args.epochs,"steps_per_epoch":args.steps,
                   "batch_size":args.batch_size,"image_size":CONFIG["image_size"],
                   "parameters":sum(p.numel() for p in model.parameters()),
                   "initialization":"random; no pretrained weights; no ResNet", "sampler":"Equal source and health class draws, replacement within each bucket",
                   "manifest_sha256":hashlib.sha256((ROOT/"data/manifest.csv").read_bytes()).hexdigest(),
                   "selection":"0.6 * mean source balanced accuracy + 0.4 * minimum source balanced accuracy; validation only",
                   "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                   "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}}
    save_json(out/"configuration.json",configuration)
    best=-1
    best_epoch=0
    history=[]
    draw_counts=np.zeros(len(rows),dtype=np.int64)
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter()
        warmup=min(1.,epoch/3)
        lr=recipe["lr"]*warmup*(.05+.95*(1+math.cos(math.pi*(epoch-1)/args.epochs))/2)
        for group in optimizer.param_groups: group["lr"]=lr
        model.train()
        loss_sum=torch.zeros((),device="cuda")
        epoch_indices=[]
        for step in range(args.steps):
            chosen=torch.cat([b[torch.randint(len(b),(args.batch_size//len(buckets),),device="cuda")] for b in buckets])
            chosen=chosen[torch.randperm(len(chosen),device="cuda")]
            epoch_indices.append(chosen)
            x=augment(images[chosen],recipe["strong"])
            y=labels[chosen]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                if recipe["mixup"] and random.random()<.5:
                    lam=float(np.random.beta(recipe["mixup"],recipe["mixup"]))
                    order=torch.randperm(len(x),device="cuda")
                    logits=model(x*lam+x[order]*(1-lam))
                    loss=lam*F.cross_entropy(logits,y,label_smoothing=.03)+(1-lam)*F.cross_entropy(logits,y[order],label_smoothing=.03)
                else:
                    loss=F.cross_entropy(model(x),y,label_smoothing=.03)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum+=loss.detach()
        draws=torch.cat(epoch_indices).cpu().numpy()
        np.add.at(draw_counts,draws,1)
        results={s:metrics(labels[idx].cpu().numpy(),probabilities(model,images,idx)) for s,idx in validation.items()}
        score=selection_score(results)
        record={"epoch":epoch,"training_loss":float(loss_sum/args.steps),"lr":lr,"selection_score":score,"validation":results,"seconds":round(time.perf_counter()-start,2)}
        history.append(record)
        if score>best:
            best=score
            best_epoch=epoch
            checkpoint={"state_dict":copy.deepcopy(model.cpu().state_dict()),"architecture":{"attention":recipe["attention"],"width":1},
                        "classes":["Healthy","Unhealthy"],"image_size":CONFIG["image_size"],"normalization":"x/127.5-1", "epoch":epoch,"validation":results,"selection_score":score,"configuration":configuration}
            torch.save(checkpoint,out/"best.pt")
            model.cuda().to(memory_format=torch.channels_last)
        save_json(out/"history.json",history)
        save_json(ROOT/"runs/progress.json",{"run":name,"epoch":epoch,"best_epoch":best_epoch,"best_score":best,"current":record})
        print(name,epoch,f"loss={record['training_loss']:.4f}"," ".join(f"{s}={r['balanced_accuracy']:.3f}" for s,r in results.items()),f"score={score:.4f} best={best:.4f} {record['seconds']}s",flush=True)
        if epoch>=20 and epoch-best_epoch>=12:
            print("Validation patience reached",flush=True)
            break
    summary={"name":name,"best_validation_score":best,"best_epoch":best_epoch,"epochs_run":epoch,"checkpoint":str(out/"best.pt"),"recipe":recipe}
    write_csv(out/"training_draws.csv",[{"index":i,"source":r["source"],"label":r["label"],"split":r["split"],"draws":int(draw_counts[i])} for i,r in enumerate(rows)])
    save_json(out/"complete.json",summary)
    del model,optimizer
    torch.cuda.empty_cache()
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=36)
    parser.add_argument("--steps",type=int,default=96)
    parser.add_argument("--batch-size",type=int,default=96)
    parser.add_argument("--seed",type=int,default=CONFIG["seed"])
    parser.add_argument("--recipes",nargs="+",choices=RECIPES,default=list(RECIPES))
    parser.add_argument("--suffix",default="")
    args=parser.parse_args()
    assert torch.cuda.is_available(),"Training requires CUDA"
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=True
    rows=read_csv(ROOT/"data/manifest.csv")
    print("Loading pixel cache onto GPU",flush=True)
    images=torch.from_numpy(np.load(ROOT/"data/images.npy")).cuda()
    results=[]
    for name in args.recipes:
        results.append(train_one(name+args.suffix,RECIPES[name],images,rows,args))
    winner=max(results,key=lambda x:x["best_validation_score"])
    save_json(ROOT/"runs/selection.json",{"runs":results,"selected":winner,"test_evaluation_done":False})
    print("Selected using validation only:",winner,flush=True)

if __name__=="__main__":
    main()
