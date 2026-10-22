"""Report-ready PNG/SVG figures and CSV tables, from saved measured results only."""
import csv
import json
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import precision_recall_curve, roc_curve
import torch

from io_utils import ROOT, read_csv, save
from model import TomatoCNN, CLASSES

FIG = ROOT / "figures"
DATASETS = {"plantvillage":("PlantVillage",ROOT/"outputs","tomato_cnn.pt"),
            "bangladesh":("Bangladesh",ROOT/"bangladesh_outputs","tomato_cnn_bangladesh.pt")}
COLORS = ["#168580","#d97935"]
plt.rcParams.update({"font.size":10,"figure.facecolor":"white","svg.fonttype":"none","svg.hashsalt":"binary-tomato-report","axes.spines.top":False,"axes.spines.right":False})


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(figure,name):
    FIG.mkdir(exist_ok=True)
    figure.savefig(FIG/f"{name}.png",dpi=200,bbox_inches="tight")
    figure.savefig(FIG/f"{name}.svg",bbox_inches="tight",metadata={"Date":None})
    svg = FIG/f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(figure)


def table(path, rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def box(ax,x,y,width,height,text,color="#e5eff5",size=10):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.012",facecolor=color,edgecolor="#31526a"))
    ax.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=size)


def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#31526a",linewidth=1.6))


def architecture():
    fig,ax=plt.subplots(figsize=(13,5.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    stages=[("RGB input\n3 × 128 × 128",.02,.90,.13),
            ("Block 1\n32 channels\n64 × 64",.20,.90,.14),("Block 2\n64 channels\n32 × 32",.39,.90,.14),
            ("Block 3\n128 channels\n16 × 16",.58,.90,.14),("Block 4\n192 channels\n8 × 8",.77,.90,.16)]
    for text,x,y,w in stages:
        box(ax,x,y-.19,w,.19,text)
    for a,b in zip(stages,stages[1:]):arrow(ax,(a[1]+a[3]+.01,.805),(b[1]-.015,.805))
    ax.text(.48,.54,"Each block:\nConv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2",ha="center",fontsize=10.5)
    lower=[("Adaptive average\npool to 2 × 2",.77,.19,.16),("Flatten\n768 values",.58,.19,.14),
           ("Dense 128 + ReLU\nDropout 0.35",.34,.19,.19),("Dense 2 + Softmax\nHealthy / Unhealthy",.02,.19,.25)]
    ax.plot([.945,.98,.98,.85],[.805,.805,.43,.43],color="#31526a",linewidth=1.6)
    arrow(ax,(.85,.43),(.85,.40))
    for text,x,y,w in lower:box(ax,x,y,w,.19,text,"#e2f2ed")
    for a,b in zip(lower,lower[1:]):arrow(ax,(a[1]-.015,.285),(b[1]+b[3]+.01,.285))
    ax.text(.5,.02,"Same custom CNN for both datasets • 939,874 trainable parameters • Two output classes",ha="center",fontsize=11)
    fig.suptitle("Tomato leaf health CNN: architecture and tensor dimensions",fontsize=15)
    export(fig,"cnn_architecture")


def workflow():
    fig,ax=plt.subplots(figsize=(13,5.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    nodes=[(.02,.65,.16,"Source images\n+ supplied labels"),(.23,.65,.16,"Audit labels, hashes\nand related images"),
           (.44,.65,.16,"Development only\n5 grouped folds"),(.66,.65,.14,"Validation recipe\nPV: fixed scratch\nBD: compare 2"),
           (.85,.65,.13,"Final fit\n20 epochs")]
    for x,y,w,text in nodes:box(ax,x,y,w,.23,text)
    for a,b in zip(nodes,nodes[1:]):arrow(ax,(a[0]+a[2]+.01,.765),(b[0]-.015,.765))
    box(ax,.23,.16,.24,.23,"Reserved test images\nNever used for recipe selection",color="#fff0dd")
    box(ax,.61,.16,.19,.23,"Freeze weights\nFixed argmax\nEvaluate test",color="#e2f2ed")
    box(ax,.85,.16,.13,.23,"Metrics, CIs\nplots + CSV",color="#e2f2ed")
    arrow(ax,(.31,.63),(.35,.41));arrow(ax,(.48,.275),(.595,.275))
    arrow(ax,(.915,.63),(.72,.41));arrow(ax,(.815,.275),(.835,.275))
    ax.text(.5,.03,"PV = PlantVillage; BD = Bangladesh. Bangladesh also reports the full released holdout with conflicting labels separately.",ha="center",fontsize=9)
    fig.suptitle("Experiment workflow: validation chooses the recipe; test measures the frozen model",fontsize=14)
    export(fig,"experiment_workflow")


def distributions():
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for ax,(key,(title,out,_)) in zip(axes,DATASETS.items()):
        audit=load(out/"data_audit.json");counts=audit["binary_counts"]
        names=["Development","Primary test"]
        splits=["development","test"]
        if key=="bangladesh":names.append("Full holdout*");splits.append("released_holdout")
        x=np.arange(len(names))
        for c,label in enumerate(CLASSES):
            vals=[counts[label][s] for s in splits]
            bars=ax.bar(x+(c-.5)*.36,vals,.36,label=label,color=COLORS[c]);ax.bar_label(bars,padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=names,ylabel="Images",title=title);ax.legend();ax.margins(y=.18)
    fig.suptitle("Dataset composition (*Bangladesh full holdout includes uncertain labels)")
    export(fig,"dataset_counts")
    audit=load(ROOT/"bangladesh_outputs/data_audit.json")
    reasons=audit["exclusions_by_reason"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    ax=axes[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.2,.76,.6,.15,"1,030 raw images")
    box(ax,.02,.40,.43,.23,"Released train: 870\nExclude 69\nDevelopment: 801","#e2f2ed")
    box(ax,.55,.40,.43,.23,"Released val: 160\nExclude 21\nPrimary test: 139","#fff0dd")
    arrow(ax,(.4,.74),(.23,.64));arrow(ax,(.6,.74),(.76,.64))
    ax.text(.5,.20,"All 160 holdout images additionally scored\nagainst uncertain filename labels",ha="center",fontsize=10)
    axes[1].barh(["Filename/annotation conflict","Missing annotation","Invalid zero-area box"],
                 [reasons.get("Filename and annotation classes disagree or image has mixed health labels",0),
                  reasons.get("Missing annotation; filename label cannot be cross-checked",0),reasons.get("Invalid/zero-area annotation box",0)],color="#bf6653")
    for bar in axes[1].containers:axes[1].bar_label(bar,padding=3)
    axes[1].set(xlabel="Excluded images",title="Bangladesh label audit");axes[1].margins(x=.15)
    fig.suptitle("Bangladesh eligibility policy was fixed before training or test predictions")
    export(fig,"bangladesh_label_audit")


def matrix_plot(key,result,normalized=False,suffix=""):
    matrix=np.array(result["confusion_matrix"])
    shown=matrix/matrix.sum(1,keepdims=True) if normalized else matrix
    fig,ax=plt.subplots(figsize=(5.8,5),constrained_layout=True)
    ax.imshow(shown,cmap="Blues",vmin=0,vmax=1 if normalized else max(matrix.max(),1))
    for i in range(2):
        for j in range(2):
            value=f"{shown[i,j]:.1%}\n(n={matrix[i,j]})" if normalized else str(matrix[i,j])
            ax.text(j,i,value,ha="center",va="center",fontsize=18,color="white" if shown[i,j]>(.5 if normalized else matrix.max()/2) else "black")
    ax.set_xticks([0,1],CLASSES);ax.set_yticks([0,1],CLASSES)
    title=DATASETS[key][0]+(" — full holdout (uncertain labels)" if suffix else " — primary test")
    ax.set(xlabel="Predicted class",ylabel="True class",title=title+("\nRow-normalized" if normalized else "\nImage counts"))
    export(fig,key+suffix+("_confusion_normalized" if normalized else "_confusion_counts"))


def dataset_figures(key,title,out,filename):
    result=load(out/"test_metrics.json");cv=load(out/"cross_validation.json")
    rows=read_csv(out/"test_predictions.csv")
    matrix_plot(key,result);matrix_plot(key,result,True)
    if key.startswith("bangladesh"):matrix_plot(key,load(out/"released_holdout_metrics.json"),suffix="_full_holdout")
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for mode in cv:
        histories=[load(out/f"{mode}_fold{f}_history.json") for f in range(1,6)]
        for ax,metric in zip(axes,("train_loss","balanced_accuracy","macro_f1")):
            values=np.array([[r[metric] for r in h] for h in histories]);x=np.arange(1,values.shape[1]+1)
            ax.plot(x,values.mean(0),label=mode);ax.fill_between(x,values.mean(0)-values.std(0),values.mean(0)+values.std(0),alpha=.15)
    for ax,label in zip(axes,("Training loss","Validation balanced accuracy","Validation macro F1")):
        ax.set(title=label,xlabel="Epoch");ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in axes[1:]:ax.set_ylim(0,1.02)
    fig.suptitle(title+": mean ± SD across five grouped validation folds")
    export(fig,key+"_cv_learning_curves")
    final=load(out/"final_history.json")
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot([h["epoch"] for h in final],[h["train_loss"] for h in final],color=COLORS[0])
    axes[1].plot([h["epoch"] for h in final],[h["train_augmented_accuracy"] for h in final],color=COLORS[1])
    axes[0].set(title="Final-fit training loss",ylabel="Training objective",xlabel="Epoch")
    axes[1].set(title="Final-fit augmented training accuracy",ylabel="Accuracy",xlabel="Epoch",ylim=(0,1.02))
    fig.suptitle(title+": final development fit (these are not validation or test curves)")
    export(fig,key+"_final_training_curves")
    actual=np.array([r["true_class"]=="Unhealthy" for r in rows]);probs=np.array([float(r["probability_Unhealthy"]) for r in rows])
    fpr,tpr,roc_threshold=roc_curve(actual,probs);precision,recall,pr_threshold=precision_recall_curve(actual,probs)
    table(out/"roc_points.csv",[{"false_positive_rate":a,"true_positive_rate":b,"threshold":c} for a,b,c in zip(fpr,tpr,roc_threshold)])
    table(out/"precision_recall_points.csv",[{"precision":a,"recall":b,"threshold":pr_threshold[i] if i<len(pr_threshold) else ""} for i,(a,b) in enumerate(zip(precision,recall))])
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    axes[0].plot(fpr,tpr,label=f"ROC AUC {result['roc_auc']:.4f}",color=COLORS[0]);axes[0].plot([0,1],[0,1],"--",color="grey",label="Chance")
    axes[1].plot(recall,precision,label=f"Average precision {result['average_precision']:.4f}",color=COLORS[1]);axes[1].axhline(actual.mean(),linestyle="--",color="grey",label="Unhealthy prevalence")
    axes[0].set(title="Unhealthy ROC",xlabel="False positive rate",ylabel="True positive rate")
    axes[1].set(title="Unhealthy precision–recall",xlabel="Recall",ylabel="Precision")
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle(title+": primary-test score curves, Unhealthy is the positive class")
    export(fig,key+"_roc_precision_recall")
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    x=np.arange(3)
    for c,label in enumerate(CLASSES):
        scores=result["per_class"][label];values=[scores[k] for k in ("precision","recall","f1-score")]
        bars=ax.bar(x+(c-.5)*.36,values,.36,label=label,color=COLORS[c]);ax.bar_label(bars,fmt="%.3f",padding=3)
    ax.set(xticks=x,xticklabels=["Precision","Recall","F1"],ylim=(0,1.25),ylabel="Score",title=title+": primary-test class metrics");ax.legend(loc="upper center",ncol=2,frameon=False)
    export(fig,key+"_class_metrics")
    keys=["accuracy","balanced_accuracy","macro_f1","recall_unhealthy","specificity_healthy"]
    values=[result["accuracy"],result["balanced_accuracy"],result["macro_f1"],result["recall_sensitivity_unhealthy"],result["specificity_healthy"]]
    bounds=np.array([result["confidence_intervals"][k+"_95"] for k in keys])
    fig,ax=plt.subplots(figsize=(8.5,4),constrained_layout=True)
    ax.errorbar(values,np.arange(5),xerr=[np.maximum(0,np.array(values)-bounds[:,0]),np.maximum(0,bounds[:,1]-np.array(values))],fmt="o",capsize=4,color=COLORS[0])
    ax.set(yticks=np.arange(5),yticklabels=["Accuracy","Balanced accuracy","Macro F1","Unhealthy recall","Healthy recall"],xlim=(0,1.02),xlabel="Score",title=title+": 95% group-bootstrap intervals")
    ax.grid(axis="x",alpha=.2)
    export(fig,key+"_confidence_intervals")
    checkpoint=torch.load(out/filename,map_location="cpu",weights_only=True)
    kernels=checkpoint["state_dict"]["features.0.weight"].numpy()
    fig,axes=plt.subplots(4,8,figsize=(10,5.3),constrained_layout=True)
    for i,(ax,kernel) in enumerate(zip(axes.flat,kernels)):
        image=kernel.transpose(1,2,0);image=(image-image.min())/(image.max()-image.min()+1e-12)
        ax.imshow(image,interpolation="nearest");ax.set_title(str(i+1),fontsize=8);ax.axis("off")
    fig.suptitle(title+": learned first-layer RGB 3×3 filters\nEach filter normalized independently for visibility; not prediction explanations")
    export(fig,key+"_first_layer_filters")
    if (out/"test_examples.jpg").exists():shutil.copyfile(out/"test_examples.jpg",FIG/f"{key}_test_examples.jpg")
    wrong=[r for r in rows if r["true_class"]!=r["predicted_class"]][:6]
    if wrong and all((ROOT/r["path"]).exists() for r in wrong):
        fig,axes=plt.subplots(math.ceil(len(wrong)/3),3,figsize=(10,3.6*math.ceil(len(wrong)/3)),squeeze=False,constrained_layout=True)
        for ax in axes.flat:ax.axis("off")
        for ax,row in zip(axes.flat,wrong):
            with Image.open(ROOT/row["path"]) as image:ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            ax.set_title(f"True: {row['true_class']} | Pred: {row['predicted_class']}",fontsize=9)
        fig.suptitle(title+": first up to six primary-test errors in recorded order")
        fig.savefig(FIG/f"{key}_errors.jpg",dpi=160,bbox_inches="tight");plt.close(fig)
    class_rows=[{"dataset":key,"class":label,"support":int(result["per_class"][label]["support"]),**{k:result["per_class"][label][k] for k in ("precision","recall","f1-score")}} for label in CLASSES]
    table(out/"class_metrics.csv",class_rows)
    table(out/"cv_summary.csv",[{"initialization":mode,**{k:v for k,v in value.items() if k.startswith("mean_") or k.startswith("sd_")}} for mode,value in cv.items()])
    table(out/"confidence_intervals.csv",[{"metric":key,"estimate":value,"lower_95":bound[0],"upper_95":bound[1]} for key,value,bound in zip(keys,values,bounds)])


def reports():
    summary=[]
    for key,(title,out,_) in DATASETS.items():
        for cohort,file in [("primary","test_metrics.json")]+([("full_holdout_uncertain_labels","released_holdout_metrics.json")] if key=="bangladesh" else []):
            result=load(out/file)
            summary.append({"dataset":key,"cohort":cohort,"test_images":result["test_images"],**{k:result[k] for k in ("accuracy","balanced_accuracy","macro_f1","precision_unhealthy","recall_sensitivity_unhealthy","specificity_healthy","roc_auc","average_precision","weighted_f1")},
                            "healthy_precision":result["per_class"]["Healthy"]["precision"],"healthy_f1":result["per_class"]["Healthy"]["f1-score"],"unhealthy_f1":result["f1_unhealthy"]})
    table(ROOT/"RESULTS.csv",summary)
    lines=["# Measured binary tomato CNN results", "", "All scores below come from saved predictions of the two completed models. Unhealthy includes the source disease/pest/damage categories. These classifiers do not identify a specific disease.", "",
           "| Dataset / cohort | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:lines.append(f"| {r['dataset']} / {r['cohort']} | {r['test_images']:,} | {r['accuracy']:.2%} | {r['balanced_accuracy']:.2%} | {r['macro_f1']:.2%} | {r['specificity_healthy']:.2%} | {r['recall_sensitivity_unhealthy']:.2%} |")
    lines += ["", "These cohorts differ and must not be treated as a controlled ranking of real-world performance. PlantVillage uses controlled images and a previously inspected benchmark split. Bangladesh's primary score applies to 139 predefined images whose supplied filename and annotation labels agree; the full 160-image holdout includes 21 contradictory labels and is only a descriptive check against uncertain filename labels. Neither model has established performance in new gardens.", "",
              "## Architecture and study design", "", "![CNN architecture](figures/cnn_architecture.png)", "", "![Experiment workflow](figures/experiment_workflow.png)", "", "![Image counts](figures/dataset_counts.png)", "", "![Bangladesh source-label audit](figures/bangladesh_label_audit.png)", "",
              "## Reusable report material", "", "Every numerical chart is available as both PNG and editable SVG in figures/. RESULTS.csv contains fractional scores, not percentages. Each result folder also contains class_metrics.csv, cv_summary.csv, confidence_intervals.csv, roc_points.csv, precision_recall_points.csv, and image-level prediction CSVs. Use figures/README.md for suggested captions and interpretation.", ""]
    for key,(title,out,_) in DATASETS.items():
        r=load(out/"test_metrics.json");matrix=r["confusion_matrix"]
        section=[f"## {title}", "", f"The primary test recognized **{matrix[0][0]} / {sum(matrix[0])} Healthy** and **{matrix[1][1]} / {sum(matrix[1])} Unhealthy** leaves.", "",
                 "| Metric | Value |", "|---|---:|"]
        for metric,label in (("accuracy","Accuracy"),("balanced_accuracy","Balanced accuracy"),("macro_f1","Macro F1"),("weighted_f1","Weighted F1"),("precision_unhealthy","Unhealthy precision"),("recall_sensitivity_unhealthy","Unhealthy recall"),("specificity_healthy","Healthy recall"),("f1_unhealthy","Unhealthy F1")):
            section.append(f"| {label} | {r[metric]:.2%} |")
        section += [f"| Healthy precision | {r['per_class']['Healthy']['precision']:.2%} |",f"| Healthy F1 | {r['per_class']['Healthy']['f1-score']:.2%} |",f"| ROC AUC | {r['roc_auc']:.4f} |",f"| Average precision | {r['average_precision']:.4f} |", ""]
        for suffix,caption in [("confusion_counts","Confusion counts"),("confusion_normalized","Row-normalized confusion matrix"),("cv_learning_curves","Grouped validation learning curves"),("final_training_curves","Final-fit training curves"),("roc_precision_recall","ROC and precision–recall"),("class_metrics","Per-class precision, recall, F1"),("confidence_intervals","Group-bootstrap uncertainty"),("first_layer_filters","Actual learned first-layer filters")]:
            section += [f"![{caption}](figures/{key}_{suffix}.png)", ""]
        section += [f"![Fixed random test examples](figures/{key}_test_examples.jpg)", ""]
        if (FIG/f"{key}_errors.jpg").exists():section += [f"![Recorded test errors](figures/{key}_errors.jpg)", ""]
        if key=="bangladesh":section += ["![Full holdout: uncertain labels](figures/bangladesh_full_holdout_confusion_counts.png)", ""]
        lines += section
        own="\n".join([f"# {title}: binary results", "", "See ../README.md for setup, ../DATA_SOURCES.md for provenance, and ../RESULTS.md for the combined report.", ""]+section)
        own=own.replace("](figures/", "](../figures/")
        (out/"REPORT.md").write_text(own.rstrip()+"\n",encoding="utf-8")
    lines += ["## Interpretation and limits", "",
              "Healthy recall is the fraction of healthy leaves recognized as healthy (also called specificity when Unhealthy is the positive class). Unhealthy recall is sensitivity. Macro F1 gives both classes equal weight; weighted F1 and overall accuracy can obscure a minority-class failure.", "",
              "95% intervals use 2,000 bootstrap resamples of the recorded image/similarity groups. These groups have incomplete plant/photo-session metadata. The intervals do not capture all label, source-grouping, or deployment uncertainty; a degenerate interval from no observed class errors does not prove perfect future recall.", "",
              "Both models use the same 939,874-parameter custom CNN at 128×128 RGB, [-1,1] normalization, and fixed argmax decisions. PlantVillage uses class-weighted CE and a fixed scratch recipe with five CV estimates. Bangladesh compares scratch against PlantVillage feature adaptation using mean final-epoch CV balanced accuracy, then refits the selected scratch recipe. Each CV fit uses 10 epochs; final fits use 20. Bangladesh uses exactly 32/32 class draws per training batch, without additional class weights. Full training configurations and artifact import provenance are in each experiment.json."]
    lines += ["", "## Additional sequential-training experiment", "", "See [FINETUNING_RESULTS.md](FINETUNING_RESULTS.md) for the full PlantVillage binary CNN fine-tuned on Bangladesh, its comparison against scratch, and the reused-holdout/unequal-budget limitations. The original two experiments above retain their recorded evidence."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    captions=["# Reusable figures and suggested captions", "", "PNG files are rendered at 200 dpi; matching SVG files retain editable text. JPG galleries use the dataset's source images; retain dataset attribution when reusing them. All curves/counts use actual saved results, without new training or test inference.", "",
              "| Figure | Suggested caption / scope |", "|---|---|",
              "| cnn_architecture | Shared binary CNN architecture and tensor dimensions; 939,874 parameters. |",
              "| experiment_workflow | Development-only grouped validation, final fitting, and reserved-test evaluation. |",
              "| dataset_counts | Healthy/Unhealthy image counts in each experiment; Bangladesh full holdout contains uncertain labels. |",
              "| bangladesh_label_audit | Predetermined label-consistency filtering and released-split counts; agreement is not an expert diagnosis. |",
              "| *_confusion_counts | Primary-test confusion counts; rows are reference classes, columns predictions. |",
              "| *_confusion_normalized | Confusion matrix normalized by reference-class support, highlighting each class's recall. |",
              "| *_cv_learning_curves | Mean ± SD across five grouped development folds; SD describes fold variation, not a confidence interval. |",
              "| *_final_training_curves | Final-fit training loss and augmented training accuracy; not held-out performance. |",
              "| *_roc_precision_recall | ROC and precision–recall curves with Unhealthy as the positive class. |",
              "| *_class_metrics | Primary-test precision, recall, and F1 for each class. |",
              "| *_confidence_intervals | 95% percentile group-bootstrap intervals; source/label uncertainty is not fully captured. |",
              "| *_first_layer_filters | Actual 3×3 RGB first-convolution filters, independently normalized; not explanations of individual predictions. |",
              "| *_test_examples.jpg | Fixed random examples selected with seed 42, not selected for correctness. |",
              "| *_errors.jpg | First up to six recorded primary-test errors, illustrating failures rather than typical performance. |",
              "| bangladesh_full_holdout_confusion_counts | Full 160-image publisher holdout against filename labels; 21 labels conflict with annotations. |", "",
              "Reproduce figures with python make_report.py. Most figures require only the tracked results and checkpoints. Error galleries can be regenerated when source images are downloaded; existing galleries are preserved otherwise."]
    captions += ["", "## PlantVillage to Bangladesh fine-tuning figures", "", "Files prefixed bangladesh_finetuned_ belong to the additional full-model fine-tuning experiment. The comparison chart uses the same Bangladesh primary holdout for both models. The workflow figure shows sequential pretraining/fine-tuning; other suffixes follow the captions above. These are exploratory reused-holdout results, with a different epoch budget from archived scratch. See [FINETUNING_RESULTS.md](../FINETUNING_RESULTS.md). Regenerate these figures separately with python report_finetune.py."]
    (FIG/"README.md").write_text("\n".join(captions)+"\n",encoding="utf-8")


def main():
    torch.set_num_threads(6)
    architecture();workflow();distributions()
    for key,(title,out,filename) in DATASETS.items():dataset_figures(key,title,out,filename)
    reports()
    print("Generated report figures (PNG/SVG), CSV tables, and results reports from saved predictions.")
