"""Fetch public dataset metadata; does not change any remote repository."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "TomatoCNN-research"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)

if __name__ == "__main__":
    out = ROOT / "data" / "sources"
    out.mkdir(parents=True, exist_ok=True)
    tree = fetch("https://api.github.com/repos/mamta-joshi-gehlot/Tomato-Village/git/trees/49da676c27fa3879ef38850a816ec74a5072f52a?recursive=1")
    (out / "tomatovillage_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    print("Items", len(tree.get("tree", [])), "truncated", tree.get("truncated"))
    from collections import Counter
    print(Counter("/".join(x["path"].split("/")[:3]) for x in tree.get("tree", [])).most_common(40))
    print("Total image MiB", sum(x.get("size",0) for x in tree.get("tree", []))/1024**2)
