"""Download and verify Bangladesh real-world tomato dataset, Mendeley V1."""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/bangladesh"
API = "https://data.mendeley.com/public-api/datasets/rnbsw72zb5"


def request(url, start=None, end=None):
    headers = {"User-Agent": "TomatoCNN-research"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end or ''}"
    return urllib.request.Request(url, headers=headers)


def parallel_download(details, partial):
    size = details["size"]
    part_dir = DATA / "download_parts"
    part_dir.mkdir(exist_ok=True)
    chunk = 16 * 1024**2
    ranges = [(start, min(start+chunk, size)-1) for start in range(0,size,chunk)]

    def fetch(bounds):
        start,end=bounds
        part=part_dir/f"{start:012d}.part"
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request(details["download_url"],start,end), timeout=45) as response:
                    assert response.status == 206, "Server must support byte-range downloads"
                    assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/"), response.headers["Content-Range"]
                    with part.open("wb") as stream:
                        while block := response.read(1024**2):
                            stream.write(block)
                assert part.stat().st_size == end-start+1
                return part
            except Exception as error:
                print(f"Range {start}-{end}, attempt {attempt+1}: {error}",flush=True)
                if attempt == 4:
                    raise
                time.sleep(1)
    done=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,bounds) for bounds in ranges]
        for future in as_completed(futures):
            done += future.result().stat().st_size
            print(f"Verified range size; total {done/1024**2:.0f}/{size/1024**2:.0f} MiB",flush=True)
    with partial.open("wb") as output:
        for start,end in ranges:
            with (part_dir/f"{start:012d}.part").open("rb") as stream:
                while block := stream.read(1024**2):
                    output.write(block)
    assert partial.stat().st_size == size


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    source_path = DATA / "source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(request(API), timeout=45) as response:
            source = json.load(response)
        source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    assert source["version"] == 1
    files = [f for f in source["files"] if f["filename"].lower().endswith(".zip")]
    assert len(files) == 1
    file = files[0]
    details = file["content_details"]
    path = DATA / file["filename"]
    expected = details["sha256_hash"]
    if not path.exists() or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != expected:
        partial = path.with_suffix(".partial")
        parallel_download(details, partial)
        with partial.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, "Publisher checksum mismatch"
        partial.replace(path)
    entries=[]
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        print(f"Verified archive checksum and CRC; {len(archive.infolist())} members", flush=True)
        for member in archive.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".txt", ".json", ".yaml", ".csv"}:
                continue
            kind = "images" if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} else "annotations"
            # Hash source paths rather than extracting archive names as filesystem paths.
            local = f"{kind}/{hashlib.sha1(member.filename.encode()).hexdigest()}{suffix}"
            destination = DATA / local
            content = archive.read(member)
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                destination.write_bytes(content)
            assert destination.read_bytes() == content
            entries.append({"source_member": member.filename, "kind": kind, "local_path": local,
                            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    (DATA / "files.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Indexed {sum(e['kind']=='images' for e in entries)} images and {sum(e['kind']=='annotations' for e in entries)} annotation/metadata files", flush=True)
    print("First archive members:", [e["source_member"] for e in entries[:15]], flush=True)


if __name__ == "__main__":
    main()
