"""Restore the exact frozen web sample; verify content hashes, without inference."""
import hashlib
import os
from pathlib import Path
import ssl
import subprocess
import urllib.error

from collect_internet_sample import get
from evaluate_finetuned import OUT
from io_utils import ROOT, read_csv


def main():
    for row in read_csv(OUT/"internet_selection.csv"):
        path=(ROOT/row["path"]).resolve()
        # Data may be junctioned to this task's original project on the same computer.
        data_root=(ROOT/"data").resolve()
        assert path.is_relative_to(data_root)
        expected=row["sha256"]
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,"Existing source file changed"
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+".download")
        try:
            temporary.write_bytes(get(row["image_url"]))
        except urllib.error.URLError as error:
            if os.name!="nt" or not isinstance(error.reason,ssl.SSLCertVerificationError):raise
            # Windows' native HTTPS validation can use the machine's certificate store.
            # Values travel as environment data, never as interpolated shell code.
            environment={**os.environ,"TOMATO_IMAGE_URL":row["image_url"],"TOMATO_IMAGE_PATH":str(temporary)}
            subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",
                            "Invoke-WebRequest -Uri $env:TOMATO_IMAGE_URL -OutFile $env:TOMATO_IMAGE_PATH -ErrorAction Stop"],
                           env=environment,check=True)
        assert hashlib.sha256(temporary.read_bytes()).hexdigest()==expected,"Publisher file changed; preserve the frozen evaluation"
        temporary.rename(path)
    print("All 16 frozen sample files are present and match their recorded SHA-256 hashes.")


if __name__=="__main__":main()
