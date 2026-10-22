"""Frozen-model checks on the reserved PlantVillage split and a fixed web sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from io_utils import ROOT, read_csv, save, sha256
from metrics import bootstrap, evaluate
from model import CLASSES, TomatoCNN, prepare_batch
from training import write_predictions

MODEL = ROOT/"bangladesh_finetuned_outputs/tomato_cnn_finetuned.pt"
OUT = ROOT/"bangladesh_finetuned_outputs/cross_dataset"


@torch.inference_mode()
def run(rows, name):
    output=OUT/name
    output.mkdir(parents=True,exist_ok=True)
    model_hash=sha256(MODEL)
    if (output/"test_metrics.json").exists():
        saved=__import__("json").loads((output/"test_metrics.json").read_text(encoding="utf-8"))
        assert saved["checkpoint_sha256"]==model_hash
        print(saved,flush=True)
        return saved
    checkpoint=torch.load(MODEL,map_location="cpu",weights_only=True)
    assert checkpoint["classes"]==CLASSES and checkpoint["image_size"]==128
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=TomatoCNN(checkpoint["channels"],2).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)

    def load(row):
        content=(ROOT/row["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest()==row["sha256"],row["path"]
        with Image.open(io.BytesIO(content)) as opened:
            image=ImageOps.exif_transpose(opened).convert("RGB")
            assert hashlib.sha256(image.tobytes()).hexdigest()==row["pixel_sha256"]
            return np.asarray(image.resize((128,128),Image.Resampling.BILINEAR))
    print(f"Loading {len(rows)} {name} images; frozen weights, {device} float32",flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        arrays=np.stack(list(pool.map(load,rows))).transpose(0,3,1,2).copy()
    images=torch.from_numpy(arrays).to(device);del arrays
    probabilities=[]
    for start in range(0,len(rows),32):
        ids=np.arange(start,min(start+32,len(rows)))
        probabilities.append(model(prepare_batch(images,ids,False)).softmax(1).cpu().numpy())
    probabilities=np.concatenate(probabilities)
    labels=np.array([CLASSES.index(r["binary_class"]) for r in rows])
    groups=np.array([r["group"] for r in rows])
    result=evaluate(labels,probabilities)
    result.update(test_images=len(rows),healthy_images=int(sum(labels==0)),unhealthy_images=int(sum(labels==1)),
                  checkpoint_sha256=model_hash,confidence_intervals=bootstrap(labels,probabilities.argmax(1),groups),
                  always_unhealthy_accuracy=float(np.mean(labels==1)),
                  evaluated_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=f"Frozen complete PV-to-Bangladesh model; {device} float32 without TF32/AMP; original EXIF/RGB/128px preprocessing; fixed argmax; no weight/threshold updates",
                  test_status="Previously inspected reserved benchmark, not new independent field data" if name=="plantvillage" else "Small convenience sample, source/manual visual labels; illustrative only, not internet-wide accuracy")
    ids=np.arange(len(rows))
    write_predictions(output/"test_predictions.csv",rows,ids,labels,probabilities)
    from prepare_bangladesh import write_rows
    write_rows(output/"evaluation_manifest.csv",rows)
    assert sha256(MODEL)==model_hash
    save(output/"test_metrics.json",result)
    print(result,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["plantvillage","internet"],required=True)
    args=parser.parse_args()
    if args.dataset=="plantvillage":
        rows=[r for r in read_csv(ROOT/"outputs/split_manifest.csv") if r["split"]=="test"]
        assert len(rows)==3632
        prior=read_csv(ROOT/"outputs/test_predictions.csv")
        assert {r["path"] for r in rows}=={r["path"] for r in prior}
        pv_development=[r for r in read_csv(ROOT/"outputs/split_manifest.csv") if r["split"]=="development"]
        bd_development=[r for r in read_csv(ROOT/"bangladesh_finetuned_outputs/split_manifest.csv") if r["split"]=="development"]
        for field in ("sha256","pixel_sha256"):
            assert not {r[field] for r in rows} & {r[field] for r in pv_development+bd_development}
    else:
        rows=read_csv(OUT/"internet_selection.csv")
        assert all(r["binary_class"] in CLASSES for r in rows)
        assert (OUT/"internet_selection.json").exists(),"Freeze labels/selection before predictions"
    run(rows,args.dataset)


if __name__=="__main__":
    main()
