"""Download only original tomato RGB images from the author's PlantVillage archive."""
import io
import json
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
SIZE = 2184723441


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.cache = OrderedDict()
        self.block_size = 8 * 1024 * 1024
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def read(self, count=-1):
        count = min(SIZE - self.position, count if count >= 0 else SIZE - self.position)
        parts = []
        while count > 0:
            start = self.position // self.block_size * self.block_size
            if start not in self.cache:
                end = min(start + self.block_size, SIZE) - 1
                request = urllib.request.Request(SOURCE + f"?download=true&block={start}", headers={"Range": f"bytes={start}-{end}"})
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(request, timeout=120) as response:
                            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                                raise RuntimeError("Server did not honor byte range")
                            block = response.read()
                        if len(block) != end - start + 1:
                            raise RuntimeError("Incomplete range download")
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                self.cache[start] = block
                self.downloaded += len(block)
                print(f"Downloaded {self.downloaded / 1024**2:.1f} MiB of selected archive ranges", flush=True)
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
            block = self.cache[start]
            local = self.position - start
            piece = block[local:local + count]
            parts.append(piece)
            self.position += len(piece)
            count -= len(piece)
        return b"".join(parts)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = DATA / "leaf-map.json"
    if not metadata.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf-map.json", metadata)
    if (DATA / "download_complete.json").exists():
        print("Original tomato images already downloaded", flush=True)
        return
    with zipfile.ZipFile(RangeReader()) as archive:
        selected = sorted([info for info in archive.infolist() if info.filename.startswith("raw/color/Tomato___") and not info.is_dir()], key=lambda x: x.header_offset)
        if len(selected) != 18160:
            raise RuntimeError(f"Expected 18160 original tomato images, got {len(selected)}")
        print(f"Archive contains {len(selected)} original tomato RGB images", flush=True)
        for i, info in enumerate(selected):
            destination = DATA / info.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(archive.read(info))
            if (i + 1) % 1000 == 0:
                print(f"Extracted {i + 1}/{len(selected)}", flush=True)
    (DATA / "download_complete.json").write_text(json.dumps({"source": SOURCE, "images": len(selected), "archive_size": SIZE}, indent=2))


if __name__ == "__main__":
    main()
