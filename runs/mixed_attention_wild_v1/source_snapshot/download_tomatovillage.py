"""Download the original multiclass subset; verify every Git blob checksum."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
TREE = ROOT / "data/sources/tomatovillage_tree.json"

def main():
    items = [x for x in json.loads(TREE.read_text())["tree"]
             if x["path"].lower().endswith((".jpg", ".png", ".jpeg"))]
    def download(item):
        target = ROOT / "data/tomatovillage" / item["path"]
        def valid(b):
            return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest() == item["sha"]
        if target.exists() and valid(target.read_bytes()):
            return
        url = "https://raw.githubusercontent.com/mamta-joshi-gehlot/Tomato-Village/main/" + urllib.parse.quote("Variant-a(Multiclass Classification)/" + item["path"])
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
                assert valid(data), item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, f in enumerate(as_completed([pool.submit(download, item) for item in items]), 1):
            f.result()
            if n % 250 == 0:
                print(f"Verified {n}/{len(items)} images", flush=True)
    print(f"Downloaded and verified {len(items)} field images", flush=True)

if __name__ == "__main__":
    main()
