"""Record additional extension-photo sources; images stay in ignored local data."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from urllib.parse import urljoin, urlparse, unquote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from io_utils import ROOT, save
from prepare_bangladesh import write_rows
from collect_internet_sample import DATA, OUT

SOURCES=[
    ("umd_key.html","https://extension.umd.edu/resource/key-common-problems-tomatoes",[
        ("HGIC-veg-tomato-Iron-Lady-early-blight-lesions-1000.jpg","Early blight on foliage","J. Traunfeld, University of Maryland Extension"),
        ("HGIC-tomato-cold injury-600x496.jpg","Cold injury to foliage","J. Traunfeld, University of Maryland Extension"),
        ("hgic-veg-septoria-leaf-spot-600.jpg","Septoria on foliage","J. Traunfeld, University of Maryland Extension"),
        ("HGIC-veg-tomato-early-blight-600.jpg","Early blight on leaves","University of Maryland Extension; see original photo credit"),
        ("HGIC-veg-tomato-bacterial-speck-600.jpg","Bacterial speck on foliage","Gerald Holmes, Strawberry Center, Cal Poly San Luis Obispo, Bugwood.org"),
        ("HGIC-veg-tomato-spider-mite-injury-384.jpg","Spider-mite feeding damage","University of Maryland Extension Ask Extension"),
        ("HGIC-veg-tomato-fusarium-5606104-576.jpg","Fusarium wilt symptoms","University of Maryland Extension / Bugwood.org; see original photo credit"),
    ]),
    ("umd_bacterial.html","https://extension.umd.edu/resource/bacterial-diseases-tomato",[
        ("hgic_veg_bacterial spot_tomato_1234099_1200.jpg","Bacterial spot on foliage","Clemson University - USDA Cooperative Extension Slide Series, Bugwood.org"),
        ("HGIC-vegetables-bacterial-leaf-spot-tomato-leaf-600.jpg","Bacterial spot on garden foliage","Florida Division of Plant Industry, Florida Department of Agriculture and Consumer Services, Bugwood.org"),
        ("hgic_veg_bacterial speck_1577155_600.jpg","Bacterial speck on a leaf","Gerald Holmes, Strawberry Center, Cal Poly San Luis Obispo, Bugwood.org"),
    ]),
]


class Images(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if tag=="img" and attrs.get("src"):
            self.urls.append(urljoin("https://extension.umd.edu",attrs["src"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    rows=[]
    for html_file,source,files in SOURCES:
        images=Images();images.feed((DATA/html_file).read_text(encoding="utf-8"))
        for basename,description,artist in files:
            matches=[u for u in images.urls if unquote(urlparse(u).path).rsplit("/",1)[-1]==basename]
            assert matches,basename
            i=32+len(rows)
            path=DATA/f"candidate_{i:02d}_publisher.jpg"
            row={"candidate_id":i,"title":description,"source_url":source,"image_url":matches[0],
                 "original_image_url":matches[0],"download_transform":"Publisher-provided image; no manual crop",
                 "artist":artist,"license":"Publisher/Bugwood photo terms apply; not redistributed in this repository",
                 "license_url":source,"source_description":description,
                 "source_revision_timestamp":"","source_file_sha1":"","path":path.relative_to(ROOT).as_posix()}
            if not args.prepare:
                with Image.open(path) as opened:
                    image=ImageOps.exif_transpose(opened).convert("RGB")
                    row.update(width=image.width,height=image.height,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                row["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
    if args.prepare:
        save(OUT/"extension_downloads.json",rows)
        print("Recorded ten extension image URLs; no predictions performed.")
        return
    candidates=json.loads((OUT/"internet_candidates.json").read_text(encoding="utf-8"))
    candidates["candidates"]=[r for r in candidates["candidates"] if int(r["candidate_id"])<32]+rows
    candidates["selection_scope"]="Available Commons photos plus ten source-labelled UMD extension examples; fixed before model inference. Wikimedia rate limits prevented further downloads. This is a convenience pool, not an unbiased internet sample."
    save(OUT/"internet_candidates.json",candidates)
    write_rows(OUT/"internet_candidates.csv",candidates["candidates"])
    fig,axes=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
    for ax in axes.flat:ax.axis("off")
    for ax,row in zip(axes.flat,rows):
        with Image.open(ROOT/row["path"]) as opened:ax.imshow(ImageOps.exif_transpose(opened).convert("RGB"))
        ax.set_title(f"{row['candidate_id']}: {row['title']}",fontsize=9)
    fig.suptitle("Extension source-label review BEFORE model predictions")
    fig.savefig(DATA/"review_extension.jpg",dpi=150);plt.close(fig)
    print("Added source-labelled photos and pre-prediction review sheet.")


if __name__=="__main__":main()
