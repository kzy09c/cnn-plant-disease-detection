"""Independent checks of saved binary predictions, splits, CV summaries, and weights."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from io_utils import ROOT, read_csv, save, sha256
from model import CLASSES, TomatoCNN
from report_assets import DATASETS, load


def verify(dataset, output=None):
    fine_tuned = dataset == "bangladesh-finetuned"
    if fine_tuned:
        dataset = "bangladesh"
        default,filename = ROOT/"bangladesh_finetuned_outputs","tomato_cnn_finetuned.pt"
    else:
        _,default,filename = DATASETS[dataset]
    out = Path(output) if output is not None else default
    result = load(out/"test_metrics.json")
    manifest = read_csv(out/"split_manifest.csv")
    by_path = {r["path"]:r for r in manifest}
    assert len(by_path) == len(manifest)
    development = [r for r in manifest if r["split"] == "development"]
    primary = [r for r in manifest if r["split"] == "test"]
    released = [r for r in manifest if r.get("source_split") == "val"] if dataset == "bangladesh" else primary
    for field in ("group","pixel_sha256","sha256"):
        assert not {r[field] for r in development} & {r[field] for r in released}
    for fold in range(1,6):
        train = [r for r in development if int(r["cv_validation_fold"]) != fold]
        validation = [r for r in development if int(r["cv_validation_fold"]) == fold]
        assert {r["binary_class"] for r in train} == {r["binary_class"] for r in validation} == set(CLASSES)
        for field in ("group","pixel_sha256","sha256"):
            assert not {r[field] for r in train} & {r[field] for r in validation}
    for row in development+primary:
        assert row["binary_class"] in CLASSES
        if dataset == "plantvillage":
            assert row["binary_class"] == ("Healthy" if row["source_class"] == "Tomato___healthy" else "Unhealthy")
        else:
            assert not row["exclusion"] and row["annotation_classes"] == str(CLASSES.index(row["binary_class"]))
    cohorts=[("test_predictions.csv","test_metrics.json",primary)]
    if dataset == "bangladesh":cohorts.append(("released_holdout_predictions.csv","released_holdout_metrics.json",released))
    for prediction_file,metric_file,expected in cohorts:
        rows=read_csv(out/prediction_file);scores=load(out/metric_file)
        assert len(rows)==len(expected)==scores["test_images"]
        assert {r["path"] for r in rows}=={r["path"] for r in expected}
        matrix=np.zeros((2,2),dtype=int)
        for row in rows:
            assert row["true_class"]==by_path[row["path"]]["binary_class"]
            probabilities=np.array([float(row["probability_Healthy"]),float(row["probability_Unhealthy"])])
            assert np.all((probabilities>=0)&(probabilities<=1)) and abs(probabilities.sum()-1)<1e-6
            assert CLASSES[probabilities.argmax()]==row["predicted_class"]
            matrix[CLASSES.index(row["true_class"]),CLASSES.index(row["predicted_class"])]+=1
        assert matrix.tolist()==scores["confusion_matrix"]
        diagonal,actual,calls=matrix.diagonal(),matrix.sum(1),matrix.sum(0)
        measured={"accuracy":matrix.trace()/matrix.sum(),"balanced_accuracy":(diagonal/actual).mean(),
                  "macro_f1":(2*diagonal/(actual+calls)).mean(),"specificity_healthy":diagonal[0]/actual[0],
                  "recall_sensitivity_unhealthy":diagonal[1]/actual[1]}
        for key,value in measured.items():assert abs(value-scores[key])<1e-10,key
        for c,label in enumerate(CLASSES):
            precision=diagonal[c]/calls[c] if calls[c] else 0
            for key,value in (("precision",precision),("recall",diagonal[c]/actual[c]),("f1-score",2*diagonal[c]/(actual[c]+calls[c]))):
                assert abs(value-scores["per_class"][label][key])<1e-10
    if dataset=="bangladesh":
        full={r["path"]:r for r in read_csv(out/"released_holdout_predictions.csv")}
        for row in read_csv(out/"test_predictions.csv"):
            for key in ("true_class","predicted_class","probability_Healthy","probability_Unhealthy"):
                assert row[key]==full[row["path"]][key]
    cv=load(out/"cross_validation.json")
    arguments=load(out/"experiment.json").get("arguments",{})
    for mode,scores in cv.items():
        histories=[load(out/f"{mode}_fold{fold}_history.json") for fold in range(1,6)]
        assert all(len(h)==arguments.get("cv_epochs",10) for h in histories)
        for key in ("accuracy","balanced_accuracy","macro_f1"):
            assert abs(np.mean([h[-1][key] for h in histories])-scores["mean_"+key])<1e-10
    assert len(load(out/"final_history.json"))==arguments.get("epochs",20)
    if dataset=="bangladesh":
        selected=max(cv,key=lambda key:(cv[key]["mean_balanced_accuracy"],cv[key]["mean_macro_f1"]))
        assert selected==result["selected_initialization"]
    model_path=out/filename
    assert sha256(model_path)==result["checkpoint_sha256"]
    checkpoint=torch.load(model_path,map_location="cpu",weights_only=True)
    assert checkpoint["classes"]==CLASSES and checkpoint["image_size"]==128
    model=TomatoCNN(checkpoint["channels"],2);model.load_state_dict(checkpoint["state_dict"])
    assert sum(p.numel() for p in model.parameters())==939874
    verified={"passed":True,"dataset":dataset,"binary_checkpoint_loaded_on_cpu":True,"checkpoint_sha256":sha256(model_path),
              "test_csv_confusion_metrics_independently_verified":True,"all_class_precision_recall_f1_verified":True,
              "split_and_cv_groups_pixels_files_disjoint":True,"cv_summaries_and_history_lengths_checked":True,
              "test_inference_performed":False}
    if fine_tuned:
        verify_transfer(out,manifest,checkpoint,result)
        verified.update(dataset="bangladesh-finetuned",full_pretrained_checkpoint_provenance_checked=True,
                        every_training_draw_checked_against_development_folds=True,all_batches_class_balanced=True)
    save(out/"verification.json",verified)
    return verified


def verify_transfer(out,manifest,checkpoint,result):
    from transfer_config import RECIPES
    record=load(out/"experiment.json")
    source=ROOT/"outputs/tomato_cnn.pt"
    assert sha256(source)==record["pretrained_sha256"]==checkpoint["pretrained_sha256"]
    assert sha256(out/"split_manifest.csv")==record["manifest_sha256"]
    assert record["recipes"]==RECIPES
    assert checkpoint["binary_head_retained"] and checkpoint["dataset"]=="plantvillage_then_bangladesh"
    original=read_csv(ROOT/"bangladesh_outputs/split_manifest.csv")
    for current,previous in zip(manifest,original,strict=True):
        for key in ("path","group","split","binary_class","cv_validation_fold","exclusion","sha256","pixel_sha256"):
            assert current.get(key,"")==previous.get(key,"")
    pv=torch.load(source,map_location="cpu",weights_only=True)
    for key in ("features.0.weight","classifier.2.weight","classifier.5.weight"):
        assert not torch.equal(pv["state_dict"][key],checkpoint["state_dict"][key]),key
    selected=load(out/"selection.json")
    assert selected["selected_initialization"]==result["selected_initialization"] and not selected["test_predictions_available"]
    assert selected["selected_utc"]<load(out/"completion.json")["completed_utc"]
    for mode in (*RECIPES,"final"):
        fits=[(mode,None)] if mode=="final" else [(f"{mode}_fold{f}",f) for f in range(1,6)]
        for name,fold in fits:
            allowed={i for i,r in enumerate(manifest) if r["split"]=="development" and (fold is None or int(r["cv_validation_fold"])!=fold)}
            seen={};batches=Counter()
            for draw in read_csv(out/f"{name}_draws.csv"):
                i,epoch,batch=int(draw["index"]),int(draw["epoch"]),int(draw["batch"])
                assert i in allowed and draw["binary_class"]==manifest[i]["binary_class"]
                batches[(epoch,batch,draw["binary_class"])]+=1
                seen.setdefault(epoch,set()).add(i)
            history=load(out/f"{name}_history.json")
            assert set(seen)==set(range(1,len(history)+1))
            assert all(ids==allowed for ids in seen.values())
            assert all(n==record["arguments"]["batch_size"]//2 for n in batches.values())
            assert all((e,b,CLASSES[1-CLASSES.index(c)]) in batches for e,b,c in batches)
            assert sum(h["optimizer_steps"] for h in history)>0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["all",*DATASETS,"bangladesh-finetuned"],default="all")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output and args.dataset=="all":parser.error("Choose a single --dataset with --output")
    available=[*DATASETS]+(["bangladesh-finetuned"] if (ROOT/"bangladesh_finetuned_outputs/test_metrics.json").exists() else [])
    for key in (available if args.dataset=="all" else [args.dataset]):print(verify(key,args.output))
