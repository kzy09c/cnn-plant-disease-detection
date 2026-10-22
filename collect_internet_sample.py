"""Retrieve a fixed Commons candidate pool for review BEFORE model predictions."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from io_utils import ROOT, save
from prepare_bangladesh import write_rows

DATA=ROOT/"data/internet_evaluation"
OUT=ROOT/"bangladesh_finetuned_outputs/cross_dataset"
TITLES=[
    "Healthy tomato leaves (7871755330).jpg",
    "Folla Tomate 006eue.jpg",
    "Lycopersicon esculentum - Tomato from Wayanad (5).jpg",
    "Mr. Stripey Heirloom Tomato leaf.jpg",
    "Solanum lycopersicum 15-p.bot-lyco.escu-001.jpg",
    "Solanum lycopersicum 15-p.bot-lyco.escu-002.jpg",
    "Solanum lycopersicum 15-p.bot-lyco.escu-003.jpg",
    "Starr 060406-7549 Solanum lycopersicum var. cerasiforme.jpg",
    "Starr 080117-2163 Solanum lycopersicum var. lycopersicum.jpg",
    "Starr 080607-7266 Solanum lycopersicum var. lycopersicum.jpg",
    "Starr 080614-9730 Solanum lycopersicum var. lycopersicum.jpg",
    "Starr-090707-2300-Solanum lycopersicum-flowers and leaves-Waikapu-Maui (24850850562).jpg",
    "Tomato leaf.jpg",
    "2023-09-28 15 01 59 Leaves and flowers on a tomato plant along High Acres Avenue in the Mountainview section of Ewing Township, Mercer County, New Jersey.jpg",
    "JfSanIldefonsoAkleH8751SchoolBulacanfvf 07.JPG",
    "Starr-130320-3489-Solanum lycopersicum-flowers and leaves-Mokolea Pt Kilauea Pt NWR-Kauai (25209258905).jpg",
    "Early blight on tomato leaves (7871930010).jpg",
    "Leaf-miner-tomato.jpg",
    "Collmer leaf speck1.jpg",
    "Passalora fulva a1 (4).jpg",
    "Passalora fulva a1 (5).jpg",
    "Passalora fulva a1 (6).jpg",
    "Tomaat bladvlekkenziekte.jpg",
    "Tomate Blatt Brantfleck.jpg",
    "Tomate Mehltau Blatt.jpg",
    "TomateBlattPhytophthorainfestans.jpg",
    "Tomato Leaf Mold.jpg",
    "Tomato septoria leaf spot 3006.jpg",
    "Tomato septoria leaf spot 3025.jpg",
    "TomateBlattRostmilbe.jpg",
    "Spray injury (7871763686).jpg",
    "TomateBlattMgMangel.jpg",
]


def get(url):
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"TomatoCNN-evaluation/1.0 (educational research)"})
            with urllib.request.urlopen(request,timeout=45) as response:return response.read()
        except urllib.error.HTTPError as error:
            if attempt==3:raise
            delay=max(15,int(error.headers.get("Retry-After","15"))) if error.code==429 else 2+attempt
            if delay>60:
                raise RuntimeError(f"Host requested {delay}s backoff; stop downloads and retry later") from error
            print(f"HTTP {error.code}; waiting {delay}s before retry",flush=True)
            time.sleep(delay)
        except Exception:
            if attempt==3:raise
            time.sleep(1+attempt)


def clean(value):
    return re.sub(r"\s+"," ",html.unescape(re.sub("<[^>]+>","",value))).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only",action="store_true",help="Review already downloaded files without further media requests")
    args=parser.parse_args()
    if (OUT/"internet_selection.json").exists():
        raise FileExistsError("Selection already frozen; preserve it")
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    query={"action":"query","format":"json","prop":"imageinfo","iiprop":"url|extmetadata|timestamp|sha1","iiurlwidth":960,
           "titles":"|".join("File:"+title for title in TITLES)}
    payload=json.loads(get("https://commons.wikimedia.org/w/api.php?"+urllib.parse.urlencode(query)))
    save(OUT/"internet_source_metadata.json",payload)
    pages={page["title"].removeprefix("File:"):page for page in payload["query"]["pages"].values()}

    def fetch(item):
        i,title=item
        page=pages[title]
        assert "imageinfo" in page,title
        info=page["imageinfo"][0];meta=info["extmetadata"]
        suffix=".jpg" if title.lower().endswith((".jpg",".jpeg")) else ".png"
        path=DATA/f"candidate_{i:02d}_960{suffix}"
        download_url=info.get("thumburl",info["url"])
        if not path.exists():
            if args.available_only:return None
            time.sleep(1)
            path.write_bytes(get(download_url))
        content=path.read_bytes()
        with Image.open(io.BytesIO(content)) as opened:
            image=ImageOps.exif_transpose(opened).convert("RGB")
            width,height=image.size
            pixels=hashlib.sha256(image.tobytes()).hexdigest()
        print(f"Downloaded candidate {i:02d}: {title}",flush=True)
        return {"candidate_id":i,"title":title,"source_url":info["descriptionurl"],"image_url":download_url,
                "original_image_url":info["url"],"download_transform":"Wikimedia 960px thumbnail when original is larger; no manual crop",
                "artist":clean(meta.get("Artist",{}).get("value","")),
                "license":clean(meta.get("LicenseShortName",{}).get("value","")),
                "license_url":meta.get("LicenseUrl",{}).get("value",""),
                "source_description":clean(meta.get("ImageDescription",{}).get("value","")),
                "source_revision_timestamp":info["timestamp"],"source_file_sha1":info["sha1"],
                "path":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(content).hexdigest(),
                "pixel_sha256":pixels,"width":width,"height":height}
    rows=[r for r in map(fetch,enumerate(TITLES)) if r is not None]
    save(OUT/"internet_candidates.json",{"retrieved_utc":datetime.now(timezone.utc).isoformat(),
        "selection_scope":"Fixed convenience candidate list from Commons tomato-leaf and disease categories; photos to be visually reviewed, deduplicated and sampled before any model inference",
        "categories":["https://commons.wikimedia.org/wiki/Category:Solanum_lycopersicum_leaves",
                      "https://commons.wikimedia.org/wiki/Category:Diseases_and_disorders_of_Solanum_lycopersicum_leaves"],"candidates":rows})
    write_rows(OUT/"internet_candidates.csv",rows)
    for start in (0,16):
        fig,axes=plt.subplots(4,4,figsize=(14,13),constrained_layout=True)
        for ax,row in zip(axes.flat,rows[start:start+16]):
            with Image.open(ROOT/row["path"]) as opened:ax.imshow(ImageOps.exif_transpose(opened).convert("RGB"))
            ax.axis("off");ax.set_title(f"{row['candidate_id']:02d}: {row['title'][:42]}",fontsize=9)
        fig.suptitle("Web candidate review BEFORE prediction; IDs fixed, no model output")
        fig.savefig(DATA/f"review_{start:02d}.jpg",dpi=150);plt.close(fig)
    print(f"Saved {len(rows)} candidate photos and two pre-prediction review sheets.")


if __name__=="__main__":main()
