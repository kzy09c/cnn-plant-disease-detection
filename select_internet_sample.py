"""Freeze reviewed weak labels and a seeded web sample before any inference."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps

from collect_internet_sample import OUT
from ensure_plantvillage_fingerprints import ensure_fingerprints
from io_utils import ROOT, save, sha256
from metrics import image_hash, distances, structural_similarity
from model import CLASSES
from prepare_bangladesh import write_rows

HEALTHY={0,1,2,3,4,8,9,10,13}
EXCLUDED={5:"Same source plant/photo series as candidate 4; keep its first view only",
          6:"Same source plant/photo series as candidate 4; keep its first view only",
          7:"Strong colour cast/possible dried sample; health reference is uncertain",
          11:"Distant foliage does not permit a defensible visual health reference",
          12:"Backlighting/yellow appearance makes the health reference uncertain",
          14:"Visible discoloration/possible damage without an explicit diagnosis; uncertain label",
          15:"Curled foliage without a health diagnosis; uncertain label",
          41:"Same bacterial-speck photograph as candidate 36 with different framing; keep first view"}


def main():
    if (OUT/"internet_selection.json").exists():
        raise FileExistsError("Selection is already frozen; do not change labels after predictions")
    rows=json.loads((OUT/"internet_candidates.json").read_text(encoding="utf-8"))["candidates"]
    assert not (OUT/"internet/test_metrics.json").exists()
    with np.load(ensure_fingerprints()) as cache:
        pv={key:cache[key] for key in ("hashes","pixels","thumbnails")}
    with np.load(ROOT/"data/bangladesh/decoded_128.npz") as cache:
        bd={key:cache[key] for key in ("hashes","pixels","thumbnails")}
    hashes=[];thumbnails=[]
    for row in rows:
        i=int(row["candidate_id"])
        row["review_exclusion"]=EXCLUDED.get(i,"")
        if i in HEALTHY:
            row["binary_class"]="Healthy"
            row["reference_basis"]="Source title says healthy; visually apparently asymptomatic" if i==0 else "Manual visual assessment: apparently asymptomatic; health unconfirmed"
        elif i==16 or i>=32:
            row["binary_class"]="Unhealthy"
            row["reference_basis"]="Source identifies disease/damage; visible symptoms reviewed"
        else:
            row["binary_class"]="Unscored"
            row["reference_basis"]="Insufficient evidence for health label"
        row["group"]=f"web_source_{i:02d}"
        if i in (0,16):row["group"]="web_sipler_20120823"
        row["overlap_exclusion"]=""
        with Image.open(ROOT/row["path"]) as opened:
            image=ImageOps.exif_transpose(opened).convert("RGB")
            assert hashlib.sha256(image.tobytes()).hexdigest()==row["pixel_sha256"]
            fingerprint=image_hash(image)
            thumbnail=np.asarray(image.convert("L").resize((64,64),Image.Resampling.LANCZOS))
        hashes.append(fingerprint);thumbnails.append(thumbnail)
        for title,reference in (("PlantVillage",pv),("Bangladesh",bd)):
            d=distances(fingerprint[None],reference["hashes"])[0]
            row["nearest_"+title.lower()+"_phash_distance"]=int(d.min())
            confirmed=any(structural_similarity(thumbnail,reference["thumbnails"][j])>=.97 for j in np.flatnonzero(d<=4))
            if row["pixel_sha256"] in set(reference["pixels"].tolist()) or confirmed:
                row["overlap_exclusion"] += f"Confirmed exact/perceptual {title} overlap; "
    hashes=np.stack(hashes)
    pairs=[]
    for i,row in enumerate(rows):
        for j in np.flatnonzero(distances(hashes[i:i+1],hashes[:i])[0]<=4) if i else []:
            similarity=structural_similarity(thumbnails[i],thumbnails[j])
            if similarity>=.97:
                row["review_exclusion"] += " Duplicate/near-duplicate of earlier candidate "+str(rows[j]["candidate_id"])
                pairs.append({"a":row["candidate_id"],"b":rows[j]["candidate_id"],"ssim":similarity})
    eligible=[r for r in rows if r["binary_class"] in CLASSES and not r["review_exclusion"] and not r["overlap_exclusion"]]
    pools=[[r for r in eligible if r["binary_class"]==label] for label in CLASSES]
    count=min(8,*map(len,pools))
    assert count>=4,"Insufficient eligible examples; report availability without inventing reference labels"
    seed=20261022
    rng=np.random.default_rng(seed)
    selected=[]
    for pool in pools:
        selected.extend(pool[int(i)] for i in rng.choice(len(pool),count,replace=False))
    selected=sorted(selected,key=lambda r:int(r["candidate_id"]))
    write_rows(OUT/"internet_review.csv",rows)
    write_rows(OUT/"internet_selection.csv",selected)
    save(OUT/"internet_overlap_audit.json",{"phash_max_distance":4,"minimum_confirmatory_ssim":.97,
        "plantvillage_reference_images":len(pv["pixels"]),"bangladesh_reference_images":len(bd["pixels"]),
        "within_web_confirmed_pairs":pairs,"overlap_exclusions":[r["candidate_id"] for r in rows if r["overlap_exclusion"]],
        "limitation":"Resizing/cropping and missing source/plant IDs can evade this conservative check"})
    save(OUT/"internet_selection.json",{"selected_utc":datetime.now(timezone.utc).isoformat(),
        "model_inference_performed":False,"seed":seed,"candidate_photos":len(rows),
        "eligible_by_class":dict(zip(CLASSES,map(len,pools))),"sampled_per_class":count,
        "selected_candidate_ids":[r["candidate_id"] for r in selected],"manifest_sha256":sha256(OUT/"internet_selection.csv"),
        "sampling":"Uniform draw without replacement separately within each visually reviewed class; at most 8 per class; fixed before inference",
        "reference_quality":"Weak labels: apparently healthy by visual review (one source-labelled healthy), disease/damage references based on publisher captions and visible symptoms. No independent diagnostic verification.",
        "selection_scope":"Convenience pool from available Commons tomato leaves plus UMD extension disease/damage photos. Random only within this pool; not a random sample of all internet images. Downloads were limited by source availability/rate limits.",
        "content_scope":"Whole photographs, no image-specific crop; mixed single/multiple leaves; Unhealthy includes biotic disease, pest damage and abiotic injury"})
    print("Frozen selected IDs:",[r["candidate_id"] for r in selected],"Eligible class counts:",list(map(len,pools)))
    print("Overlap exclusions:",[r["candidate_id"] for r in rows if r["overlap_exclusion"]])


if __name__=="__main__":main()
