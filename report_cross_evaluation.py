"""Verify and plot frozen fine-tuned model evaluations from saved prediction CSVs."""
import csv
import json

import numpy as np
from sklearn.metrics import confusion_matrix

from io_utils import ROOT, read_csv, save, sha256
from metrics import evaluate
from model import CLASSES
import report_assets as charts
from evaluate_finetuned import MODEL, OUT


def verify(name):
    folder=OUT/name
    result=charts.load(folder/"test_metrics.json")
    rows=read_csv(folder/"evaluation_manifest.csv")
    predictions=read_csv(folder/"test_predictions.csv")
    by_path={r["path"]:r for r in rows}
    assert len(by_path)==len(rows)==len(predictions)==result["test_images"]
    assert {r["path"] for r in predictions}==set(by_path)
    assert sha256(MODEL)==result["checkpoint_sha256"]
    actual=[];scores=[]
    for row in predictions:
        assert row["path"] in by_path
        assert row["true_class"]==by_path[row["path"]]["binary_class"]
        actual.append(CLASSES.index(row["true_class"]))
        probability=[float(row["probability_Healthy"]),float(row["probability_Unhealthy"])]
        assert all(0<=p<=1 for p in probability) and abs(sum(probability)-1)<1e-6
        assert CLASSES[np.argmax(probability)]==row["predicted_class"]
        scores.append(probability)
    actual=np.array(actual);scores=np.array(scores,dtype=np.float32)
    measured=evaluate(actual,scores)
    for key in ("accuracy","balanced_accuracy","macro_f1","weighted_f1","precision_unhealthy","recall_sensitivity_unhealthy","specificity_healthy","roc_auc","average_precision"):
        assert abs(result[key]-measured[key])<1e-7,key
    matrix=np.array(result["confusion_matrix"])
    assert matrix.tolist()==measured["confusion_matrix"]
    # Independent confusion-count arithmetic, separate from sklearn evaluation.
    assert abs(matrix.trace()/matrix.sum()-result["accuracy"])<1e-12
    assert abs(np.mean(matrix.diagonal()/matrix.sum(1))-result["balanced_accuracy"])<1e-12
    assert abs(np.mean(2*matrix.diagonal()/(matrix.sum(0)+matrix.sum(1)))-result["macro_f1"])<1e-12
    if name=="plantvillage":
        original=[r for r in read_csv(ROOT/"outputs/split_manifest.csv") if r["split"]=="test"]
        assert {r["path"] for r in rows}=={r["path"] for r in original}
        training=[r for r in read_csv(ROOT/"outputs/split_manifest.csv") if r["split"]=="development"]
        training += [r for r in read_csv(ROOT/"bangladesh_finetuned_outputs/split_manifest.csv") if r["split"]=="development"]
        for field in ("sha256","pixel_sha256"):
            assert not {r[field] for r in rows} & {r[field] for r in training}
    else:
        selected=read_csv(OUT/"internet_selection.csv")
        assert {r["path"] for r in selected}==set(by_path)
        selection=charts.load(OUT/"internet_selection.json")
        assert sha256(OUT/"internet_selection.csv")==selection["manifest_sha256"]
        assert selection["selected_utc"]<result["evaluated_utc"]
        assert selection["model_inference_performed"] is False
        assert all(r["overlap_exclusion"]=="" for r in rows)
    verified={"passed":True,"checkpoint_unchanged":True,"CSV_metrics_verified":True,
              "population":"Original reserved PlantVillage test split" if name=="plantvillage" else "Fixed, pre-labelled internet convenience sample",
              "test_inference_during_verification":False}
    save(folder/"verification.json",verified)
    print(name,verified)
    return result


def plot(name,result):
    import matplotlib.pyplot as plt
    key="finetuned_cross_"+name
    title="Fine-tuned model on PlantVillage" if name=="plantvillage" else "Fine-tuned model on internet sample (illustrative)"
    charts.DATASETS[key]=(title,OUT/name,"unused")
    charts.matrix_plot(key,result);charts.matrix_plot(key,result,True)
    charts.table(OUT/name/"class_metrics.csv",[{"class":label,"support":int(result["per_class"][label]["support"]),
                  **{k:result["per_class"][label][k] for k in ("precision","recall","f1-score")}} for label in CLASSES])
    fig,ax=plt.subplots(figsize=(8.5,4.7),constrained_layout=True)
    x=np.arange(3)
    for c,label in enumerate(CLASSES):
        values=[result["per_class"][label][k] for k in ("precision","recall","f1-score")]
        bars=ax.bar(x+(c-.5)*.36,values,.36,label=label,color=charts.COLORS[c]);ax.bar_label(bars,fmt="%.3f",padding=3)
    ax.set(xticks=x,xticklabels=["Precision","Recall","F1"],ylim=(0,1.25),ylabel="Score",title=title)
    ax.legend(loc="upper center",ncol=2,frameon=False)
    charts.export(fig,key+"_class_metrics")


def main():
    results={name:verify(name) for name in ("plantvillage","internet")}
    for name,result in results.items():plot(name,result)
    summary=[]
    bd=charts.load(ROOT/"bangladesh_finetuned_outputs/test_metrics.json")
    for name,result in (("Bangladesh primary (reused)",bd),("PlantVillage test (reused)",results["plantvillage"]),("Internet convenience sample (weak labels)",results["internet"])):
        summary.append({"dataset":name,"images":result["test_images"],**{k:result[k] for k in ("accuracy","balanced_accuracy","macro_f1","specificity_healthy","recall_sensitivity_unhealthy")}})
    charts.table(ROOT/"CROSS_DATASET_RESULTS.csv",summary)
    pv,web=results["plantvillage"],results["internet"]
    original=charts.load(ROOT/"outputs/test_metrics.json")
    selection=charts.load(OUT/"internet_selection.json")
    lines=["# Frozen PlantVillage → Bangladesh model: additional testing","",
           "The same completed fine-tuned binary model is evaluated unchanged on PlantVillage's reserved test split and an additional internet-photo sample. No new training, threshold tuning, image-specific crop or model selection was performed. Inputs use the recorded EXIF orientation, RGB, 128×128 resize and [-1,1] normalization; prediction is float32 with fixed argmax.","",
           "| Evaluation population | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['images']} | {r['accuracy']:.2%} | {r['balanced_accuracy']:.2%} | {r['macro_f1']:.2%} | {r['specificity_healthy']:.2%} | {r['recall_sensitivity_unhealthy']:.2%} |")
    lines += ["","## PlantVillage retention check","",
              f"The unchanged original test split contains 3,632 images: 318 Healthy and 3,314 Unhealthy. All test file and decoded-pixel hashes were checked against the original manifest, and their hashes are disjoint from both stages' development images. This is a previously inspected controlled-image benchmark; it is not independent field validation.","",
              f"Before Bangladesh fine-tuning, the archived PlantVillage model scored **{original['accuracy']:.2%} accuracy / {original['macro_f1']:.2%} macro F1** on this split. After fine-tuning, the new checkpoint scored **{pv['accuracy']:.2%} / {pv['macro_f1']:.2%}**. Its PlantVillage performance dropped substantially. This measures reduced retention of the original domain; it does not identify which weight/statistic changes caused the drop.","",
              f"An always-Unhealthy classifier would get **{pv['always_unhealthy_accuracy']:.2%} accuracy** on this imbalanced split, but only 50% balanced accuracy. The fine-tuned model recognized **{pv['confusion_matrix'][0][0]}/318 Healthy** and **{pv['confusion_matrix'][1][1]}/3,314 Unhealthy**. Accuracy alone would obscure this behaviour.","",
              "![PlantVillage confusion](figures/finetuned_cross_plantvillage_confusion_counts.png)","",
              "![PlantVillage class metrics](figures/finetuned_cross_plantvillage_class_metrics.png)","","## Internet-photo check: scope and label quality","",
              f"The final sample contains **{web['test_images']} images**, with **{web['healthy_images']} apparently Healthy and {web['unhealthy_images']} Unhealthy**. Candidates came from [Wikimedia Commons tomato-leaf categories](https://commons.wikimedia.org/wiki/Category:Solanum_lycopersicum_leaves) and University of Maryland Extension's [tomato problem guide](https://extension.umd.edu/resource/key-common-problems-tomatoes) / [bacterial-disease guide](https://extension.umd.edu/resource/bacterial-diseases-tomato), rather than from an unbiased sample of the entire internet. After reviewing image content and source descriptions and excluding overlaps/duplicate views, a reproducible seeded draw selected the fixed sample before inference (seed {selection['seed']}). It is random only within that reviewed convenience pool. Host rate limits prevented further Commons image downloads; that availability limit is recorded, and no predictions were used to select sources.","",
              "The Healthy references are predominantly **visual assessments of apparently asymptomatic foliage**, not confirmed absence of disease. Unhealthy references use source captions/descriptions plus visible symptoms; they include pest damage and noninfectious stress in keeping with the broad binary task. These are weak reference labels, so accuracy and F1 below are illustrative agreement with those labels, not validated diagnostic accuracy or a claim about arbitrary internet images. Some photos show multiple leaves/backgrounds, providing an additional framing shift.","",
              "Source descriptions, authors, available file revision identifiers, download URLs, attribution/terms, image hashes, pre-prediction review notes, exclusions and the final sample are retained in the cross_dataset directory. Wikimedia 960-pixel thumbnails are used when originals are larger; extension photos use the publisher-provided image size. There is no manual leaf cropping. Conservative exact-pixel/perceptual checks against all PlantVillage and Bangladesh images identified no retained matches at the recorded cutoff; incomplete source metadata cannot establish total independence.","",
              f"The model recognized **{web['confusion_matrix'][0][0]}/{web['healthy_images']} apparently Healthy** and **{web['confusion_matrix'][1][1]}/{web['unhealthy_images']} Unhealthy** photos. Small-sample uncertainty and label errors limit interpretation. Images with uncertain health/content were excluded before predictions, rather than removed because the model got them wrong.","",
              "![Internet confusion](figures/finetuned_cross_internet_confusion_counts.png)","",
              "![Internet class metrics](figures/finetuned_cross_internet_class_metrics.png)","","## Individual internet predictions","",
              "| Source image | Reference | Prediction | Unhealthy softmax | Reference basis |","|---|---|---|---:|---|"]
    by_path={r["path"]:r for r in read_csv(OUT/"internet_selection.csv")}
    for prediction in read_csv(OUT/"internet/test_predictions.csv"):
        source=by_path[prediction["path"]]
        lines.append(f"| [{source['title']}]({source['source_url']}) | {prediction['true_class']} | {prediction['predicted_class']} | {float(prediction['probability_Unhealthy']):.3f} | {source['reference_basis']} |")
    lines += ["","## Uncertainty and reproducibility","",
              "Each result folder includes per-image prediction CSVs, class precision/recall/F1, 2,000 group-bootstrap intervals, a frozen evaluation manifest and verification.json. Intervals do not account for all label uncertainty, missing plant/source relationships, or repeated benchmark use. Softmax scores are not guaranteed calibrated probabilities.","",
              "```powershell","# Source datasets must already be downloaded for this inference:","python evaluate_finetuned.py --dataset plantvillage","",
              "# Restore the exact selected web thumbnails from their recorded URLs:","python restore_internet_sample.py","python evaluate_finetuned.py --dataset internet","",
              "# Verify/rebuild tables and charts without running inference:","python report_cross_evaluation.py","```","",
              "Completed evaluation results are reused after checkpoint verification. Restoring files verifies their exact downloaded SHA-256 hashes and stops if a source file has changed. Raw downloaded web photos and pre-prediction review sheets are local data, not bundled with the repository; links/attribution remain available in the manifest. Original trained models and earlier result files are preserved."]
    (ROOT/"CROSS_DATASET_RESULTS.md").write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")
    print("Saved cross-dataset results, verified metrics and six PNG/SVG figure pairs.")


if __name__=="__main__":main()
