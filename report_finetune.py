"""Rebuild the transfer report and figures from saved predictions, without inference."""
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.stats import binomtest

import report_assets as charts
from io_utils import ROOT, read_csv, save
from model import CLASSES
from train_finetune import DEFAULT_OUTPUT as OUT, MODEL_FILENAME

KEY = "bangladesh_finetuned"
TITLE = "PlantVillage to Bangladesh fine-tuning"


def gallery():
    rows=read_csv(OUT/"test_predictions.csv")
    if not all((ROOT/r["path"]).exists() for r in rows):
        return
    rng=np.random.default_rng(42)
    chosen=[]
    for label in CLASSES:
        pool=[r for r in rows if r["true_class"]==label]
        chosen.extend(pool[int(i)] for i in rng.choice(len(pool),4,replace=False))
    fig,axes=plt.subplots(2,4,figsize=(12,6.5),constrained_layout=True)
    for ax,row in zip(axes.flat,chosen):
        with Image.open(ROOT/row["path"]) as opened:
            ax.imshow(ImageOps.exif_transpose(opened).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"Reference: {row['true_class']}\nPrediction: {row['predicted_class']}",fontsize=9)
    fig.suptitle("Fine-tuned model: fixed random test examples (seed 42), not selected for correctness")
    fig.savefig(OUT/"test_examples.jpg",dpi=160,bbox_inches="tight");plt.close(fig)


def paired_comparison():
    records=[]
    for cohort,predictions,metrics_file in (("primary","test_predictions.csv","test_metrics.json"),
                                            ("full_holdout_uncertain_labels","released_holdout_predictions.csv","released_holdout_metrics.json")):
        before={r["path"]:r for r in read_csv(ROOT/"bangladesh_outputs"/predictions)}
        after=read_csv(OUT/predictions)
        assert set(before)=={r["path"] for r in after}
        old=np.array([before[r["path"]]["predicted_class"]==r["true_class"] for r in after])
        new=np.array([r["predicted_class"]==r["true_class"] for r in after])
        worse=int(sum(old & ~new));better=int(sum(~old & new))
        base=charts.load(ROOT/"bangladesh_outputs"/metrics_file);fine=charts.load(OUT/metrics_file)
        records.append({"cohort":cohort,"images":len(after),"scratch_only_correct":worse,"finetuned_only_correct":better,
                        "both_correct":int(sum(old & new)),"both_wrong":int(sum(~old & ~new)),
                        "accuracy_delta":fine["accuracy"]-base["accuracy"],
                        "balanced_accuracy_delta":fine["balanced_accuracy"]-base["balanced_accuracy"],
                        "macro_f1_delta":fine["macro_f1"]-base["macro_f1"],
                        "exploratory_exact_mcnemar_p":float(binomtest(better,worse+better,.5).pvalue) if worse+better else 1.0})
    charts.table(OUT/"paired_comparison.csv",records)
    save(OUT/"paired_comparison.json",{"comparisons":records,"note":"Paired descriptive comparison on reused holdouts, not confirmatory hypothesis testing; unequal training budgets; full cohort contains conflicting labels"})
    return records


def transfer_workflow():
    fig,ax=plt.subplots(figsize=(12.5,4.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    nodes=[(.02,.60,.21,"PlantVillage development\n14,528 images\nExisting trained binary CNN"),
           (.29,.60,.20,"Copy every trained layer\nKeep Healthy/Unhealthy head\n939,874 parameters"),
           (.56,.60,.19,"Bangladesh development\n801 images; 5 folds\nChoose learning rates"),
           (.81,.60,.17,"Reload original PV model\nFine-tune all layers\n30 final epochs")]
    for x,y,w,text in nodes:charts.box(ax,x,y,w,.25,text,size=9)
    for a,b in zip(nodes,nodes[1:]):charts.arrow(ax,(a[0]+a[2]+.01,.725),(b[0]-.015,.725))
    charts.box(ax,.03,.15,.27,.23,"Full checkpoint loaded fresh for every fit\nTrain-only augmentation; balanced batches\nBatchNorm statistics also adapt",color="#e2f2ed",size=9)
    charts.box(ax,.38,.15,.25,.23,"Test cohort: previously inspected\n139 label-agreement images\nAll 160 also reported separately",color="#fff0dd",size=9)
    charts.box(ax,.71,.15,.27,.23,"Freeze final weights; fixed argmax\nOne holdout inference pass\nAccuracy, macro F1, class recall + CIs",color="#e2f2ed",size=9)
    charts.arrow(ax,(.65,.265),(.695,.265));charts.arrow(ax,(.9,.58),(.85,.40))
    fig.suptitle("Sequential training: PlantVillage first, then Bangladesh",fontsize=15)
    charts.export(fig,KEY+"_workflow")


def build_notebook():
    import hashlib
    import nbformat as nbf
    book=nbf.v4.new_notebook()
    book.metadata.kernelspec={"name":"plant-tomato-cnn","display_name":"Tomato CNN (Python 3.12)","language":"python"}
    sources=[("markdown","# PlantVillage → Bangladesh: full-model fine-tuning\n\nThe complete PlantVillage binary CNN, including its Healthy/Unhealthy head, is fine-tuned on Bangladesh development images. This notebook reviews saved results on CPU. The reused holdout is exploratory; it is not new independent evidence."),
             ("code",'from pathlib import Path\nimport json, sys\nfrom IPython.display import display, Image, Markdown\nROOT=Path.cwd()\nif not (ROOT/"model.py").exists():\n    ROOT=ROOT/"cnn-plant-disease-detection"\nsys.path.insert(0,str(ROOT))\nOUT=ROOT/"bangladesh_finetuned_outputs"\nmetrics=json.loads((OUT/"test_metrics.json").read_text(encoding="utf-8"))\nrecord=json.loads((OUT/"experiment.json").read_text(encoding="utf-8"))\nprint("Stages:", *record["stages"], sep="\\n")\nprint("Selected:", metrics["selected_initialization"])\nprint("GPU used for training:",record["gpu"])'),
             ("code",'import torch\nfrom model import TomatoCNN\ncheckpoint=torch.load(OUT/"tomato_cnn_finetuned.pt",map_location="cpu",weights_only=True)\nmodel=TomatoCNN(checkpoint["channels"],2)\nmodel.load_state_dict(checkpoint["state_dict"])\nmodel.eval()\nprint("Classes:",checkpoint["classes"])\nprint("Parameters:",sum(p.numel() for p in model.parameters()))\nprint("Original binary head retained:",checkpoint["binary_head_retained"])\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_workflow.png"),width=1100))'),
             ("code",'cv=json.loads((OUT/"cross_validation.json").read_text(encoding="utf-8"))\nfor recipe,scores in cv.items():\n    print(recipe, "CV balanced accuracy:",f"{scores[\"mean_balanced_accuracy\"]:.2%}", "CV macro F1:",f"{scores[\"mean_macro_f1\"]:.2%}")\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_cv_learning_curves.png"),width=1100))\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_final_training_curves.png"),width=1000))'),
             ("code",'for key in ("accuracy","balanced_accuracy","macro_f1","precision_unhealthy","recall_sensitivity_unhealthy","specificity_healthy","roc_auc"):\n    print(key,f"{metrics[key]:.4f}")\nprint("Reference rows / prediction columns:",metrics["confusion_matrix"])\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_comparison.png"),width=1000))\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_confusion_counts.png"),width=650))\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_roc_precision_recall.png"),width=1000))'),
             ("code",'full=json.loads((OUT/"released_holdout_metrics.json").read_text(encoding="utf-8"))\nprint("Full holdout contains 21 conflicting filename/annotation labels.")\nfor key in ("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy"):\n    print(key,f"{full[key]:.2%}")\ndisplay(Image(filename=str(ROOT/"figures/bangladesh_finetuned_full_holdout_confusion_counts.png"),width=650))\nprint(json.loads((OUT/"verification.json").read_text(encoding="utf-8")))'),
             ("markdown",'## Try the model\n\n```powershell\npython predict.py "C:\\path\\to\\leaf.jpg" --dataset bangladesh-finetuned\n```\n\nThe model predicts tomato leaf health, without naming a disease. See FINETUNING_RESULTS.md for the comparison and limitations.')]
    book.cells=[getattr(nbf.v4,"new_"+kind+"_cell")(source) for kind,source in sources]
    for cell in book.cells:cell.id=hashlib.sha1(cell.source.encode()).hexdigest()[:12]
    nbf.write(book,ROOT/"PlantVillage_Bangladesh_Finetuning.ipynb")


def main():
    import torch
    torch.set_num_threads(6)
    charts.DATASETS[KEY]=(TITLE,OUT,MODEL_FILENAME)
    gallery();charts.dataset_figures(KEY,TITLE,OUT,MODEL_FILENAME);transfer_workflow()
    comparisons=paired_comparison()
    result=charts.load(OUT/"test_metrics.json");full=charts.load(OUT/"released_holdout_metrics.json")
    base=charts.load(ROOT/"bangladesh_outputs/test_metrics.json")
    cv=charts.load(OUT/"cross_validation.json");record=charts.load(OUT/"experiment.json")
    fields=("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy")
    fig,ax=plt.subplots(figsize=(11,4.8),constrained_layout=True)
    x=np.arange(len(fields))
    for j,(label,r,color) in enumerate((("Bangladesh scratch (archived)",base,"#168580"),("PV → Bangladesh fine-tuned",result,"#8d60b8"))):
        bars=ax.bar(x+(j-.5)*.36,[r[k] for k in fields],.36,label=label,color=color);ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=["Accuracy","Balanced\naccuracy","Macro F1","Healthy\nrecall","Unhealthy\nrecall"],ylim=(0,1.23),ylabel="Score",title="Same 139-image Bangladesh primary cohort: exploratory comparison")
    ax.legend(loc="upper center",ncol=2,frameon=False)
    charts.export(fig,KEY+"_comparison")
    comparison_rows=[]
    for title,folder in (("Bangladesh scratch","bangladesh_outputs"),("PV then Bangladesh fine-tuned","bangladesh_finetuned_outputs")):
        for cohort,file in (("primary","test_metrics.json"),("full_holdout_uncertain_labels","released_holdout_metrics.json")):
            r=charts.load(ROOT/folder/file)
            comparison_rows.append({"model":title,"cohort":cohort,"images":r["test_images"],**{k:r[k] for k in fields}})
    charts.table(ROOT/"FINETUNING_RESULTS.csv",comparison_rows)
    lines=["# PlantVillage first, then Bangladesh: measured fine-tuning results","",
           "This additional model starts from the existing trained PlantVillage binary checkpoint and updates every layer on Bangladesh development data. It retains the trained Healthy/Unhealthy output head, unlike the earlier feature-adaptation experiment that reset that head. The original two models/results remain available.","",
           "## Measured comparison on the same Bangladesh images","",
           "| Model / cohort | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in comparison_rows:
        lines.append(f"| {r['model']} / {r['cohort']} | {r['images']} | {r['accuracy']:.2%} | {r['balanced_accuracy']:.2%} | {r['macro_f1']:.2%} | {r['specificity_healthy']:.2%} | {r['recall_sensitivity_unhealthy']:.2%} |")
    change=comparisons[0]
    lines += ["",f"Fine-tuning changed primary-test accuracy by **{100*change['accuracy_delta']:+.2f} percentage points** and macro F1 by **{100*change['macro_f1_delta']:+.2f} points** versus the archived scratch model. In paired predictions, {change['finetuned_only_correct']} images changed from wrong to correct and {change['scratch_only_correct']} changed from correct to wrong.","",
              "This is an exploratory comparison on a previously inspected holdout. Recipes were selected using development folds only, but the study as a whole has already inspected these test images. The new default budget is 20 CV epochs / 30 final epochs, versus 10/20 for archived scratch; these scores do not isolate the effect of initialization at equal training budget. The 139-image cohort requires supplied label agreement; the full 160-image cohort includes 21 conflicting labels. Neither proves generalization to new gardens.","",
              "## Training stages and validation selection","",f"1. Reuse the PlantVillage binary CNN trained on **14,528 development images**. Its **3,632 benchmark test images were excluded from weight training**; no new PlantVillage pretraining was needed.",
              "2. Audit source overlap and reuse the same **801 Bangladesh development images**, **139 primary test images**, released 160-image holdout and five grouped folds as the scratch experiment.",
              "3. For each fold and recipe, load the exact complete PlantVillage checkpoint afresh. Keep all layers trainable, including the output head, and update BatchNorm running statistics. Use original train-only augmentation and exactly 32 Healthy / 32 Unhealthy draws per default batch, with unweighted cross entropy.",
              "4. Choose the recipe with highest mean final-epoch validation balanced accuracy; macro F1 breaks ties. Reload PlantVillage weights afresh and train on all Bangladesh development images. Save the final weights before one pass over the released holdout. Use fixed argmax decisions.","",
              "| Fine-tuning recipe | Feature LR | Classifier LR | Mean CV balanced accuracy | Mean CV macro F1 |","|---|---:|---:|---:|---:|"]
    for recipe,scores in cv.items():
        rates=record["recipes"][recipe]
        lines.append(f"| {recipe} | {rates['features_lr']} | {rates['classifier_lr']} | {scores['mean_balanced_accuracy']:.2%} | {scores['mean_macro_f1']:.2%} |")
    lines += ["",f"Selected recipe: **{result['selected_initialization']}**. AdamW uses weight decay 0.0001 and a cosine schedule to 0.000001. Architecture remains 939,874 parameters, 128×128 RGB, [-1,1] normalization.","",
              "Full-model fine-tuning is a standard transfer-learning approach; see the [official PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). It is a reasonable experiment because both datasets involve tomato leaves, but controlled-image features can transfer poorly to natural backgrounds.","",
              "## Class metrics and uncertainty","","| Class | Precision | Recall | F1 | Support |","|---|---:|---:|---:|---:|"]
    for label in CLASSES:
        r=result["per_class"][label]
        lines.append(f"| {label} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1-score']:.2%} | {int(r['support'])} |")
    lines += ["",f"ROC AUC: **{result['roc_auc']:.4f}**. Average precision: **{result['average_precision']:.4f}**. Weighted F1: **{result['weighted_f1']:.2%}**. Unhealthy is the positive class. Raw prediction scores and all class metrics are retained as CSV files.","",
              "95% intervals use 2,000 group-bootstrap draws and appear in confidence_intervals.csv / .json. They do not capture label uncertainty, all same-plant relationships, repeated test inspection or deployment shift. Paired prediction counts and an exploratory exact McNemar statistic are in paired_comparison.csv; no confirmatory inference is claimed.","","## Figures for reuse",""]
    for suffix,caption in (("workflow","Sequential training workflow"),("comparison","Same-cohort scratch/fine-tuning comparison"),("cv_learning_curves","Mean and SD over five validation folds"),("final_training_curves","Final development training curves"),("confusion_counts","Primary confusion matrix"),("confusion_normalized","Primary class recalls"),("roc_precision_recall","Primary ROC and precision–recall curves"),("class_metrics","Class precision, recall and F1"),("confidence_intervals","Group-bootstrap intervals"),("full_holdout_confusion_counts","Full holdout with uncertain labels"),("first_layer_filters","Learned first-layer filters")):
        lines += [f"![{caption}](figures/{KEY}_{suffix}.png)",""]
    for suffix in ("test_examples","errors"):
        if (ROOT/f"figures/{KEY}_{suffix}.jpg").exists():lines += [f"![{suffix}](figures/{KEY}_{suffix}.jpg)",""]
    lines += ["## Use and reproduce","","```powershell",'python predict.py "C:\\path\\to\\leaf.jpg" --dataset bangladesh-finetuned',
              "python verify_results.py --dataset bangladesh-finetuned","python report_finetune.py",
              "python execute_notebook.py PlantVillage_Bangladesh_Finetuning.ipynb","", "# New training run; completed outputs are protected:",
              "python train_finetune.py --output runs/bangladesh-finetuned-new","```","",
              "New computers need both source datasets downloaded and the PlantVillage fingerprint cache for fresh training. Use python download_data.py, python download_bangladesh.py, and python ensure_plantvillage_fingerprints.py first. Prediction, verification and notebook review use the tracked checkpoint/results on CPU.","",
              "The checkpoint is bangladesh_finetuned_outputs/tomato_cnn_finetuned.pt. All new evidence is stored separately in bangladesh_finetuned_outputs/. Dataset attribution and label/grouping limitations remain in DATA_SOURCES.md and BANGLADESH_LICENSE.txt. Rebuilding the report never runs inference; rebuilding the notebook source clears its displayed outputs, which can be refreshed with execute_notebook.py."]
    lines += ["", "## Additional testing of this frozen checkpoint", "", "See [CROSS_DATASET_RESULTS.md](CROSS_DATASET_RESULTS.md) for its later PlantVillage retention check and fixed internet-photo convenience sample. Those results preserve the weights and the original Bangladesh evidence; the small web sample has weak reference labels."]
    report="\n".join(lines).rstrip()+"\n"
    (ROOT/"FINETUNING_RESULTS.md").write_text(report,encoding="utf-8")
    (OUT/"REPORT.md").write_text(report.replace("](figures/","](../figures/").replace("](CROSS_DATASET_RESULTS.md)","](../CROSS_DATASET_RESULTS.md)"),encoding="utf-8")
    build_notebook()
    print("Generated the full-model fine-tuning report, CSV comparisons, figures and notebook source.")


if __name__=="__main__":
    main()
