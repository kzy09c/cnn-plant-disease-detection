"""Inspect the frozen Pakistan release and cache its ZIP directory for raw download."""
from pathlib import Path, PurePosixPath
from collections import Counter
from zipfile import ZipFile
import json
from prepare_data import ROOT, save_json
from inspect_remote_zip import RemoteZip

def main():
    out=ROOT/'data/3mbnb82mxd_unseen'
    out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((ROOT/'dataset_sources/pakistan/source_metadata.json').read_text(encoding='utf-8'))
    assert metadata['id']=='3mbnb82mxd' and metadata['version']==2
    file=next(f for f in metadata['files'] if f['filename']=='Tomato_Leaf_Disease_Dataset_Pakistan.zip')
    with ZipFile(RemoteZip(file['content_details']['download_url'],file['content_details']['size'])) as archive:
        records=[{'name':r.filename,'size':r.file_size,'compressed':r.compress_size,'offset':r.header_offset,'crc':r.CRC} for r in archive.infolist()]
    raw=[r for r in records if 'Dataset (raw)' in PurePosixPath(r['name']).parts and PurePosixPath(r['name']).suffix.lower() in {'.jpg','.jpeg','.png'}]
    assert len(raw)==830
    save_json(out/'official_release.json',metadata)
    save_json(out/'archive_members.json',records)
    print('Frozen release 2: raw image counts',dict(Counter(PurePosixPath(r['name']).parent.name for r in raw)))

if __name__=='__main__':main()
