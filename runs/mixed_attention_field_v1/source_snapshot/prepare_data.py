"""Combine audited legacy splits, verify bytes, and prevent duplicate leakage."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "local_config.json").read_text())
DATA = ROOT / "data"

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

def fingerprint(image):
    gray = image.convert("L")
    thumb = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS))
    freq = dctn(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float), norm="ortho")[:8, :8].ravel()
    bits = freq > np.median(freq[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), "big"), thumb

def ssim(a, b):
    x = a.astype(np.float32).reshape(8,8,8,8).swapaxes(1,2).reshape(64,64) / 255
    y = b.astype(np.float32).reshape(8,8,8,8).swapaxes(1,2).reshape(64,64) / 255
    mx,my = x.mean(1),y.mean(1)
    vx,vy = x.var(1),y.var(1)
    cv = ((x-mx[:,None])*(y-my[:,None])).mean(1)
    return float((((2*mx*my+.01**2)*(2*cv+.03**2))/((mx**2+my**2+.01**2)*(vx+vy+.03**2))).mean())

class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value,[index],{}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value,[index],{}]
                return
            node = node[2][distance]

    def query(self, value, radius=4):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from node[1]
            stack.extend(v for k,v in node[2].items() if distance-radius <= k <= distance+radius)

def source_rows():
    old = Path(CONFIG["legacy_project"])
    repo = Path(CONFIG["legacy_repository"])
    sources = [("PlantVillage",repo/"outputs/split_manifest.csv",repo),
               ("Bangladesh",repo/"bangladesh_outputs/split_manifest.csv",repo),
               ("TLID",old/"tlid_outputs/split_manifest.csv",old),
               ("PlantDoc",old/"plantdoc_binary_outputs/split_manifest.csv",old),
               ("Internet16",repo/"bangladesh_finetuned_outputs/cross_dataset/internet_selection.csv",repo)]
    rows=[]
    for source, manifest, base in sources:
        for item in read_csv(manifest):
            if item.get("split") == "excluded" or item.get("exclusion"):
                continue
            split = "test" if source == "Internet16" or item["split"] == "test" else ("validation" if item["cv_validation_fold"] == "1" else "train")
            row = dict(item)
            row.update(source=source,path=str((base/item["path"]).resolve()),
                       split=split, original_split=item.get("split","test"),
                       original_manifest=str(manifest), label=int(item["binary_class"] == "Unhealthy"),
                       category=item.get("category",item.get("source_class",item["binary_class"])))
            rows.append(row)
    return rows

def main():
    DATA.mkdir(exist_ok=True)
    rows = source_rows()
    n=len(rows)
    size=CONFIG["image_size"]
    import sys
    reuse="--reuse-cache" in sys.argv
    print(f"Verifying and caching {n} images at {size}px; reuse={reuse}",flush=True)
    pixels=None if reuse else np.lib.format.open_memmap(DATA/"images.npy", mode="w+", dtype=np.uint8, shape=(n,3,size,size))
    thumbs=np.empty((n,64,64),np.uint8)
    hashes=[]
    def load(args):
        i,row=args
        path=Path(row["path"])
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"],path
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert("RGB")
            ph,thumb=fingerprint(image)
            pixel=hashlib.sha256(image.tobytes()).hexdigest()
            small=np.asarray(image.resize((size,size),Image.Resampling.BILINEAR)).transpose(2,0,1)
        return i,ph,thumb,pixel,small
    if reuse:
        cached=read_csv(DATA/"manifest.csv")
        assert len(cached)==n
        for row, previous in zip(rows,cached):
            assert row['path']==previous['path'] and row['sha256']==previous['sha256']
            row['pixel_sha256']=previous['pixel_sha256']
            row['phash']=previous['phash']
        hashes=[int(r['phash'],16) for r in rows]
        thumbs=np.load(DATA/"thumbnails.npy")
    else:
        with ThreadPoolExecutor(max_workers=10) as pool:
            for i,ph,thumb,pixel,small in pool.map(load,enumerate(rows)):
                pixels[i]=small
                thumbs[i]=thumb
                hashes.append(ph)
                rows[i]["pixel_sha256"]=pixel
                rows[i]["phash"]=f"{ph:016x}"
                if (i+1)%5000==0:
                    print(f"Cached {i+1}/{n}",flush=True)
        pixels.flush()
        np.save(DATA/"thumbnails.npy",thumbs)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        parent[find(b)]=find(a)
    seen={}
    for i,r in enumerate(rows):
        for key in [("group",r["source"],r["group"]),("pixel",r["pixel_sha256"])]:
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    tree=BKTree()
    pairs=[]
    for i,ph in enumerate(hashes):
        for j in tree.query(ph):
            if find(i)!=find(j) and ssim(thumbs[i],thumbs[j])>=.97:
                union(i,j)
                pairs.append([i,j])
        tree.add(ph,i)
        if (i+1)%10000==0: print(f"Similarity audit {i+1}/{n}",flush=True)
    groups=defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    changes=[]
    priority={"train":0,"validation":1,"test":2}
    exact_labels=defaultdict(set)
    for row in rows: exact_labels[row['pixel_sha256']].add(row['label'])
    conflicting_pixels={pixel for pixel,labels in exact_labels.items() if len(labels)>1}
    for g,indices in groups.items():
        split=max((rows[i]["split"] for i in indices),key=priority.get)
        conflicting=any(rows[i]['pixel_sha256'] in conflicting_pixels for i in indices)
        # A plant/session group may legitimately include both healthy and damaged
        # leaves. Only identical decoded pixels with contradictory labels exclude it.
        if conflicting: split="excluded"
        for i in indices:
            if rows[i]["split"]!=split: changes.append({"index":i,"from":rows[i]["split"],"to":split})
            rows[i]["split"]=split
            rows[i]["merged_group"]=str(g)
            rows[i]["index"]=i
    write_csv(DATA/"manifest.csv",rows)
    counts=Counter((r["source"],r["split"],r["binary_class"]) for r in rows)
    audit={"counts":{" / ".join(k):v for k,v in sorted(counts.items())},
           "near_duplicate_pairs":len(pairs),"split_changes":changes,
           "method":"Original groups plus decoded duplicates and pHash<=4 with block SSIM>=.97; highest-priority split wins: test > validation > train",
           "limitations":"Incomplete parent-photo, plant and camera-session IDs. Similarity checks do not prove biological or source independence. Legacy tests were previously inspected.",
           "manifest_sha256":hashlib.sha256((DATA/"manifest.csv").read_bytes()).hexdigest()}
    save_json(DATA/"audit.json",audit)
    save_json(DATA/"similarity_pairs.json",pairs)
    print(json.dumps(audit["counts"],indent=2),flush=True)

if __name__=="__main__":
    main()
