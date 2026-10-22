"""Freeze an entire newly downloaded source as an external test dataset."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import re
import numpy as np
from PIL import Image, ImageOps
from prepare_data import ROOT, DATA, CONFIG, read_csv, write_csv, save_json, BKTree, fingerprint, ssim

def main():
    base=read_csv(DATA/"manifest.csv")
    base_thumbs=np.load(DATA/"thumbnails.npy")
    tree=BKTree()
    for i,r in enumerate(base): tree.add(int(r["phash"],16),i)
    exact={r["pixel_sha256"] for r in base}
    paths=sorted(p for p in (DATA/"tomatovillage").rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"})
    assert len(paths)==4525
    size=CONFIG['image_size']
    images=np.lib.format.open_memmap(DATA/"external_images.npy",mode="w+",dtype=np.uint8,shape=(len(paths),3,size,size))
    def load(path):
        raw=path.read_bytes()
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert("RGB")
            pixel=hashlib.sha256(image.tobytes()).hexdigest()
            ph,thumb=fingerprint(image)
            small=np.asarray(image.resize((size,size),Image.Resampling.BILINEAR)).transpose(2,0,1)
        exclusion="exact overlap with legacy data" if pixel in exact else ""
        if not exclusion:
            for j in tree.query(ph):
                if ssim(thumb,base_thumbs[j])>=.97:
                    exclusion="structural near overlap with legacy data"
                    break
        category=path.parent.name
        family=re.sub(r"_\d+$","",path.stem)
        row={"index":0,"source":"TomatoVillage","path":str(path.resolve()),"binary_class":"Healthy" if category=="Healthy" else "Unhealthy",
             "label":int(category!="Healthy"),"category":category,"sha256":hashlib.sha256(raw).hexdigest(),
             "pixel_sha256":pixel,"phash":f"{ph:016x}","group":family,"exclusion":exclusion,
             "publisher_split":path.relative_to(DATA/"tomatovillage").parts[0],"split":"external_test"}
        return row,small
    rows=[]
    with ThreadPoolExecutor(max_workers=10) as pool:
        for i,(row,small) in enumerate(pool.map(load,paths)):
            row['index']=i
            rows.append(row)
            images[i]=small
            if (i+1)%1000==0: print("External cache",i+1,flush=True)
    # Exclude entire source-photo families if any member overlaps previous data.
    excluded_groups={r['group'] for r in rows if r['exclusion']}
    for row in rows:
        if row['group'] in excluded_groups and not row['exclusion']:
            row['exclusion']='family of overlap with legacy data'
    images.flush()
    write_csv(DATA/"external_manifest.csv",rows)
    from collections import Counter
    save_json(DATA/"external_audit.json",{
        "source":"https://github.com/mamta-joshi-gehlot/Tomato-Village",
        "classification_tree":"49da676c27fa3879ef38850a816ec74a5072f52a",
        "role":"Entire source held out from training and recipe selection; all publisher splits combined for external evaluation",
        "counts":dict(Counter(r['category'] for r in rows if not r['exclusion'])),
        "excluded":dict(Counter(r['exclusion'] for r in rows if r['exclusion'])),
        "manifest_sha256":hashlib.sha256((DATA/"external_manifest.csv").read_bytes()).hexdigest(),
        "limitations":"Capture groups inferred from filenames; incomplete plant/garden IDs. External source was selected for availability, not a probability sample of the internet. Unhealthy includes nutrient deficiency and insect damage. No broad internet accuracy claim."})
    print("External dataset frozen",len(rows),"excluded",sum(bool(r['exclusion']) for r in rows),flush=True)

if __name__=="__main__":
    main()
