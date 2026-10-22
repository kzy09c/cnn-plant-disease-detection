"""Download and inspect the public 2,212-photo Tomato-Disease release."""
import json
import hashlib
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from prepare_data import ROOT,save_json

if __name__=='__main__':
    source=json.loads((ROOT/'data/sources/zenodo15868289.json').read_text())
    info=source['files'][0]
    out=ROOT/'data/tomato_disease'
    out.mkdir(exist_ok=True)
    target=out/info['key']
    print('License',source['metadata'].get('license'),'checksum',info['checksum'],flush=True)
    if not target.exists():
        chunks=out/'download_parts'
        chunks.mkdir(exist_ok=True)
        length=info['size']
        chunk_size=8*1024*1024
        offsets=list(range(0,length,chunk_size))
        def get_part(start):
            end=min(start+chunk_size,length)-1
            part=chunks/f'{start:012d}.part'
            if part.exists() and part.stat().st_size==end-start+1:return
            request=urllib.request.Request(info['links']['self'],headers={'Range':f'bytes={start}-{end}','User-Agent':'TomatoCNN-research'})
            with urllib.request.urlopen(request,timeout=45) as response:
                assert response.status==206
                data=response.read()
            assert len(data)==end-start+1
            part.write_bytes(data)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for n,future in enumerate(as_completed([pool.submit(get_part,s) for s in offsets]),1):
                future.result()
                if n%8==0: print('Downloaded parts',n,'/',len(offsets),flush=True)
        with target.with_suffix('.partial').open('wb') as f:
            for start in offsets:f.write((chunks/f'{start:012d}.part').read_bytes())
        target.with_suffix('.partial').replace(target)
    algorithm,expected=info['checksum'].split(':')
    digest=hashlib.new(algorithm)
    with target.open('rb') as f:
        while chunk:=f.read(1024*1024):digest.update(chunk)
    assert digest.hexdigest()==expected
    with ZipFile(target) as z:
        print('Archive members',len(z.namelist()),flush=True)
        print(Counter('/'.join(p.split('/')[:3]) for p in z.namelist()).most_common(25),flush=True)
        print('Metadata files',[p for p in z.namelist() if Path(p).suffix.lower() in {'.yaml','.yml','.json','.names'}],flush=True)
        names=z.namelist()
        save_json(out/'archive_members.json',names)
        for item in z.infolist():
            if item.is_dir():continue
            destination=(out/'extracted'/item.filename).resolve()
            assert destination.is_relative_to((out/'extracted').resolve()),item.filename
            if not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True)
                # Preserve original names even when Windows' legacy path limit
                # would reject the combined workspace + source filename.
                extended=Path('\\\\?\\'+str(destination)) if __import__('os').name=='nt' else destination
                extended.write_bytes(z.read(item))
    print('Downloaded, verified, extracted',flush=True)
