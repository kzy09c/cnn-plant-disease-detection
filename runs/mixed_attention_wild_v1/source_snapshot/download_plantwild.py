"""Download only official tomato members using byte ranges; verify ZIP CRCs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,struct,time,urllib.request,zlib
from prepare_data import ROOT,DATA,save_json,write_csv

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    out=DATA/'plantwild'
    source=json.loads((out/'archive_source.json').read_text())
    items=json.loads((out/'archive_index.json').read_text())
    url=source['url']
    split={}
    for line in (out/'trainval.txt').read_text().splitlines():
        path,cls,mode=line.rsplit('=',2)
        split['plantwild/images/'+path]=(cls,{'0':'test','1':'train','2':'validation'}[mode])
    urls=json.loads((out/'url_record.json').read_text())
    prompts=json.loads((out/'plantwild_prompts.json').read_text())
    print('Tomato leaf class has',len(prompts.get('tomato leaf',[])),'generated prompts; image labels come from the class taxonomy, not GPT prompts.',flush=True)
    def download(item):
        name=item['filename']
        category=Path(name).parent.name
        target=out/'images'/category/(hashlib.sha1(name.encode()).hexdigest()+Path(name).suffix.lower())
        for attempt in range(4):
            try:
                data=target.read_bytes() if target.exists() else b''
                if not (len(data)==item['file_size'] and zlib.crc32(data)&0xffffffff==item['CRC']):
                    start=item['header_offset'];end=start+30+len(name.encode())+item['extra_length']+item['compress_size']+255
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'TomatoCNN-research'})
                    with urllib.request.urlopen(req,timeout=30) as response:
                        assert response.status==206
                        body=response.read()
                    header=struct.unpack_from('<IHHHHHIIIHH',body)
                    assert header[0]==0x04034b50
                    offset=30+header[9]+header[10]
                    compressed=body[offset:offset+item['compress_size']]
                    assert len(compressed)==item['compress_size']
                    data=zlib.decompress(compressed,-15) if item['compress_type']==8 else compressed
                    assert item['compress_type'] in [0,8]
                assert len(data)==item['file_size'] and zlib.crc32(data)&0xffffffff==item['CRC']
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix('.partial');temporary.write_bytes(data);temporary.replace(target)
                cls,partition=split[name]
                return {'path':str(target.resolve()),'source_member':name,'category':category,'source':'PlantWild',
                        'binary_class':'Healthy' if category=='tomato leaf' else 'Unhealthy','label':int(category!='tomato leaf'),
                        'publisher_split':partition,'source_class_id':cls,'source_url':urls.get(name,''),
                        'sha256':hashlib.sha256(data).hexdigest(),'zip_crc32':item['CRC']}
            except Exception as error:
                if attempt==3:raise
                print('Retry',name,repr(error),flush=True);time.sleep(attempt+1)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(download,item) for item in items]
        for n,future in enumerate(as_completed(futures),1):
            rows.append(future.result())
            if n%200==0:print('Verified tomato photos',n,'/',len(items),flush=True)
    rows.sort(key=lambda x:x['source_member'])
    write_csv(out/'download_manifest.csv',rows)
    print('Downloaded',len(rows),'unmodified original tomato images',flush=True)
