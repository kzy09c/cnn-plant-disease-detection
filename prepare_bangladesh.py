"""Audit Bangladesh V1 labels/groups before training, preserving the released holdout."""
import csv
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from io_utils import read_csv, save
from model import CLASSES
from metrics import image_hash, distances, structural_similarity
from ensure_plantvillage_fingerprints import ensure_fingerprints

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/bangladesh"
OUT = ROOT / "bangladesh_outputs"


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def prepare(size=128, output=None):
    OUT = Path(output) if output is not None else ROOT / "bangladesh_outputs"
    if (OUT / "test_metrics.json").exists():
        raise FileExistsError("Completed results are protected; call train.py with a fresh --output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = DATA / "files.json"
    assert index_path.exists(), "Run download_bangladesh.py first"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    annotations = {r["source_member"]: r for r in index if r["kind"] == "annotations"}
    entries = sorted([r for r in index if r["kind"] == "images"], key=lambda r: r["source_member"])
    assert len({r["source_member"] for r in entries}) == len(entries)
    cache = DATA / f"decoded_{size}.npz"
    cache_key = hashlib.sha256((hashlib.sha256(index_path.read_bytes()).hexdigest()+str(size)).encode()).hexdigest()
    cached = None
    if cache.exists():
        with np.load(cache) as saved:
            assert str(saved["key"]) == cache_key
            cached = {key:saved[key] for key in ("arrays","hashes","thumbnails","pixels","dimensions")}
    print(f"Auditing {len(entries)} Bangladesh images and their annotations", flush=True)

    def read(item):
        i, entry = item
        member = PurePosixPath(entry["source_member"])
        match = re.fullmatch(r"([HD])\s*\((\d+)\)(?:\((\d+)\))?", member.stem)
        assert match is not None, f"Inspect unknown filename label: {member}"
        expected = 0 if match[1] == "H" else 1
        original_split = member.parent.name
        assert original_split in {"train", "val"}
        label_member = str(member).replace("/Images/", "/labels/").rsplit(".", 1)[0]+".txt"
        exclusion = []
        boxes = []
        annotation_hash = ""
        if label_member not in annotations:
            exclusion.append("Missing annotation; filename label cannot be cross-checked")
        else:
            annotation = annotations[label_member]
            content = (DATA / annotation["local_path"]).read_bytes()
            annotation_hash = hashlib.sha256(content).hexdigest()
            assert annotation_hash == annotation["sha256"]
            for line in content.decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, label_member
                category = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                assert category in (0,1) and all(np.isfinite(coordinates)), label_member
                if not all(-0.001 <= x <= 1.001 for x in coordinates) or min(coordinates[2:]) <= 0:
                    if "Invalid/zero-area annotation box" not in exclusion:
                        exclusion.append("Invalid/zero-area annotation box")
                boxes.append(category)
            if not boxes or set(boxes) != {expected}:
                exclusion.append("Filename and annotation classes disagree or image has mixed health labels")
        path = DATA / entry["local_path"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        if cached is None:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
                array = np.asarray(image.resize((size,size), Image.Resampling.BILINEAR))
                phash = image_hash(image)
                thumbnail = np.asarray(image.convert("L").resize((64,64),Image.Resampling.LANCZOS))
                dimensions = [image.width, image.height]
        else:
            pixels = str(cached["pixels"][i])
            array, phash, thumbnail = cached["arrays"][i], cached["hashes"][i], cached["thumbnails"][i]
            dimensions = cached["dimensions"][i].tolist()
        row = {"path": "data/bangladesh/"+entry["local_path"], "source_member": str(member),
               "source_split": original_split, "binary_class": CLASSES[expected], "filename_label": match[1],
               "source_id": f"{match[1]}_{match[2]}",
               "annotation_classes": ";".join(map(str,sorted(set(boxes)))), "annotation_boxes": len(boxes),
               "annotation_sha256": annotation_hash, "sha256": entry["sha256"], "pixel_sha256": pixels,
               "width": dimensions[0], "height": dimensions[1], "exclusion": " | ".join(exclusion)}
        return row, array, phash, thumbnail

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(read, enumerate(entries)))
    rows = [x[0] for x in loaded]
    arrays = np.stack([x[1] for x in loaded])
    hashes = np.stack([x[2] for x in loaded])
    thumbnails = np.stack([x[3] for x in loaded])
    if cached is None:
        np.savez_compressed(cache, key=cache_key, arrays=arrays, hashes=hashes, thumbnails=thumbnails,
                            pixels=np.array([r["pixel_sha256"] for r in rows]), dimensions=np.array([[r["width"],r["height"]] for r in rows]))
    del loaded
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a,b):
        parent[find(b)] = find(a)

    def exclude(i, reason):
        if reason not in rows[i]["exclusion"]:
            rows[i]["exclusion"] += (" | " if rows[i]["exclusion"] else "") + reason

    seen = {}
    for i,row in enumerate(rows):
        for key in (("file",row["sha256"]),("pixels",row["pixel_sha256"]),("filename_copy_family",row["source_id"])):
            if key in seen:
                union(i,seen[key])
            else:
                seen[key]=i
    pairs=[]
    for start in range(0,len(rows),64):
        matrix=distances(hashes[start:start+64],hashes)
        for offset,i in enumerate(range(start,min(start+64,len(rows)))):
            for j in np.flatnonzero(matrix[offset,:i]<=4):
                similarity=structural_similarity(thumbnails[i],thumbnails[j])
                if similarity >= .97:
                    union(i,int(j))
                    pairs.append({"a":rows[i]["source_member"],"b":rows[j]["source_member"],"phash_distance":int(matrix[offset,j]),"thumbnail_ssim":similarity})
    groups=np.array([f"bangladesh_{find(i)}" for i in range(len(rows))])
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group)
        if len({rows[i]["binary_class"] for i in ids if not rows[i]["exclusion"]})>1:
            for i in ids:
                exclude(i,"Similarity group has conflicting health labels")
    # Prevent overlap of pretrained features and this experiment's validation/test images.
    pv_cache = ensure_fingerprints()
    with np.load(pv_cache) as pv:
        pv_hashes, pv_pixels, pv_thumbnails = pv["hashes"], pv["pixels"], pv["thumbnails"]
    pv_manifest=read_csv(ROOT / "outputs/split_manifest.csv")
    assert set(pv_pixels.tolist()) == {r["pixel_sha256"] for r in pv_manifest}
    pv_set=set(pv_pixels.tolist())
    for start in range(0,len(rows),64):
        matrix=distances(hashes[start:start+64],pv_hashes)
        for offset,i in enumerate(range(start,min(start+64,len(rows)))):
            rows[i]["nearest_plantvillage_phash_distance"]=int(matrix[offset].min())
            confirmed=any(structural_similarity(thumbnails[i],pv_thumbnails[j])>=.97 for j in np.flatnonzero(matrix[offset]<=4))
            if rows[i]["pixel_sha256"] in pv_set or confirmed:
                exclude(i,"Exact/structurally confirmed overlap with PlantVillage pretraining images")
    released_test=np.array([i for i,r in enumerate(rows) if r["source_split"]=="val"])
    test_groups=set(groups[released_test])
    for i,row in enumerate(rows):
        if row["source_split"]=="train" and groups[i] in test_groups:
            exclude(i,"Training similarity group overlaps released holdout; preserve holdout")
    labels=np.array([CLASSES.index(r["binary_class"]) for r in rows])
    development=np.array([i for i,r in enumerate(rows) if r["source_split"]=="train" and not r["exclusion"]])
    test=np.array([i for i,r in enumerate(rows) if r["source_split"]=="val" and not r["exclusion"]])
    assert not set(groups[development]) & set(groups[released_test])
    assert not {rows[i]["pixel_sha256"] for i in development} & {rows[i]["pixel_sha256"] for i in released_test}
    assert set(labels[test])==set(labels[development])=={0,1}
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=43)
    folds=[]
    for fold,(a,b) in enumerate(splitter.split(development,labels[development],groups[development]),1):
        training,validation=development[a],development[b]
        assert not set(groups[training]) & set(groups[validation])
        assert set(labels[training])==set(labels[validation])=={0,1}
        folds.append((training,validation))
        for i in validation:
            rows[i]["cv_validation_fold"]=fold
    for i,row in enumerate(rows):
        row["group"]=groups[i]
        row["split"]="excluded" if row["exclusion"] else ("development" if row["source_split"]=="train" else "test")
    write_rows(OUT / "split_manifest.csv",rows)
    save(OUT / "similarity_pairs.json",pairs)
    audit={"dataset":"Bangladesh real-world tomato leaves, Mendeley V1", "doi":"10.17632/rnbsw72zb5.1",
           "source_url":"https://data.mendeley.com/datasets/rnbsw72zb5/1","license":"CC BY 4.0",
           "image_size":size,"downloaded_images":len(rows),"annotation_files":len(annotations),
           "raw_filename_counts":dict(Counter(r["binary_class"] for r in rows)),
           "development_images":len(development),"test_images":len(test),"released_holdout_images":len(released_test),
           "development_groups":len(set(groups[development])),"test_groups":len(set(groups[test])),
           "binary_counts":{label:{"development":int(sum(labels[development]==c)),"test":int(sum(labels[test]==c)),
                                   "released_holdout":int(sum(labels[released_test]==c))} for c,label in enumerate(CLASSES)},
           "exclusions_by_reason":dict(Counter(reason for row in rows for reason in row["exclusion"].split(" | ") if reason)),
           "excluded_by_source_split":dict(Counter(r["source_split"] for r in rows if r["exclusion"])),
           "label_policy":"Whole-image filename H=Healthy/D=Unhealthy; use only images whose every annotation agrees under 0=Healthy/1=Unhealthy; ID meaning inferred from dominant filename/annotation agreement, not independent diagnosis",
           "test_policy":"Released val reserved as test; only label-consistent eligible subset supplies primary score; full released val also scored descriptively against uncertain filename labels",
           "grouping":"Class-local filename copy families, exact decoded pixels and files; connected pHash<=4 AND 64x64 block SSIM>=0.97; training groups overlapping any released val removed",
           "confirmed_similarity_pairs":len(pairs),"largest_group":max(Counter(groups).values()),
           "plantvillage_pretraining_images_audited":len(pv_pixels),
           "limitations":"No reliable plant or photo-session IDs; conservative similarity grouping cannot identify all same-plant views. Agreement of two supplied labels does not establish clinical ground truth. Two gardens provide limited geographic coverage.",
           "published_count_discrepancy":"Archive has 1030 images/1028 annotation files; paper reports 1028 images (482 Healthy/546 Diseased), download page says 486 Healthy/546 Diseased with a contradictory 1028 total."}
    save(OUT / "data_audit.json",audit)
    print(json.dumps(audit,indent=2),flush=True)
    # A deterministic contact sheet makes source ambiguity reviewable without any predictions.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bad=[i for i,r in enumerate(rows) if "Filename and annotation" in r["exclusion"]]
    chosen=np.random.default_rng(42).choice(bad,min(8,len(bad)),replace=False)
    fig,axes=plt.subplots(2,4,figsize=(12,6.5),constrained_layout=True)
    for ax,i in zip(axes.flat,chosen):
        ax.imshow(arrays[i]); ax.axis("off")
        ax.set_title(f"{Path(rows[i]['source_member']).name}\nFilename: {rows[i]['binary_class']}; box IDs: {rows[i]['annotation_classes']}",fontsize=9)
    fig.suptitle("Label disagreements excluded from the primary experiment; no model predictions")
    fig.savefig(OUT / "annotation_disagreements.jpg",dpi=140)
    plt.close(fig)
    return rows, arrays.transpose(0,3,1,2).copy(), labels, groups, development, test, released_test, folds, audit


if __name__=="__main__":
    prepare()
