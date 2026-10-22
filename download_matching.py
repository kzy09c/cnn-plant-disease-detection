"""Freeze and download raw originals from an unused source with matched categories."""
from pathlib import Path, PurePosixPath
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
import binascii, hashlib, json, struct, time, urllib.request, zlib
from prepare_data import ROOT, save_json, write_csv
from inspect_remote_zip import RemoteZip

DIRECTORY=ROOT/'data/3mbnb82mxd_unseen'
MODEL_SHA='e905416007b09d826595af17fa4f5d5b15c103c973790853f707de73a38f1108'
MAPPING={
 'Tomato_Early_blight_leaf':('Early blight','Tomato___Early_blight',119),
 'Tomato_Healthy_leaf':('Healthy','Tomato___healthy',137),
 'Tomato_leaf_late_blight':('Late blight','Tomato___Late_blight',113),
 'Tomato_leaf_yellow_curl_virus':('Yellow leaf curl virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus',80),
 'Tomato_mold_leaf':('Leaf mold','Tomato___Leaf_Mold',186),
 'Tomato_Septoria_leaf_spot':('Septoria leaf spot','Tomato___Septoria_leaf_spot',195),
}

def main():
    metadata=json.loads((DIRECTORY/'official_release.json').read_text(encoding='utf-8'))
    assert metadata['id']=='3mbnb82mxd' and metadata['version']==2
    file=metadata['files'][0];url=file['content_details']['download_url']
    records=json.loads((DIRECTORY/'archive_members.json').read_text(encoding='utf-8'))
    with ZipFile(RemoteZip(url,file['content_details']['size'])) as archive:
        for name in ['Tomato Dataset/License.txt','Tomato Dataset/README.md']:
            content=archive.read(name)
            (DIRECTORY/PurePosixPath(name).name).write_bytes(content)
            print(name,content.decode('utf-8',errors='replace').encode('ascii','backslashreplace').decode(),flush=True)
    selected=[]
    for item in records:
        parts=PurePosixPath(item['name']).parts
        if len(parts)==4 and parts[1]=='Dataset (raw)' and parts[2] in MAPPING and PurePosixPath(parts[-1]).suffix.lower() in {'.jpg','.jpeg','.png'}:
            category,training_category,expected=MAPPING[parts[2]]
            selected.append({**item,'category':category,'training_category':training_category,
                             'label':int(category!='Healthy'),'binary_class':'Healthy' if category=='Healthy' else 'Unhealthy',
                             'path':str(DIRECTORY/'images'/parts[2]/parts[-1])})
    assert len(selected)==830
    assert Counter(r['category'] for r in selected)=={category:count for category,train,count in MAPPING.values()}
    assert hashlib.sha256((ROOT/'best_model.pt').read_bytes()).hexdigest()==MODEL_SHA
    protocol={'source_url':'https://data.mendeley.com/datasets/3mbnb82mxd/2','source_name':metadata['name'],
              'source_id':'3mbnb82mxd','release_version':2,'checkpoint_sha256':MODEL_SHA,'threshold':.5,
              'eligibility':'All 830 raw original images in the six named category folders. Exclude all 7200 augmented members. These source images were not deliberately used in training; audit overlaps before treating any photo as unseen.',
              'training_category_mapping':{category:train for category,train,count in MAPPING.values()},
              'expected_category_counts':{category:count for category,train,count in MAPPING.values()},
              'binary_task':'Healthy folder -> Healthy; five named disease folders -> Unhealthy. Per-disease counts measure correct binary predictions, not identification of the disease name.',
              'audit_rule':'Exclude entire families overlapping any earlier seven-source data/parent photos or the Agri-Vision4 check (decoded duplicate or pHash<=4 plus SSIM>=.97). Exclude contradictory identical-pixel references, retain one per decoded photo, group remaining structural matches.',
              'test_rule':'Existing final checkpoint only. No retraining, threshold/crop selection or label selection from predictions. Freeze eligibility before inference.',
              'provenance_caveat':'Web description says 7200 field images; archive actually contains 830 raw and 7200 augmented images. Source labels/location claims are not independently diagnosed/verified here.',
              'download_integrity':'Selective ZIP byte ranges and verified member CRC32; record file SHA256. Full archive SHA256 recorded but not verified because unselected bytes are not downloaded.',
              'publisher_archive_sha256':file['content_details']['sha256_hash'],'license':'CC BY 4.0 according to release page and archived licence.'}
    save_json(DIRECTORY/'protocol_before_predictions.json',protocol)
    save_json(DIRECTORY/'selected_members.json',selected)
    ordered=sorted(records,key=lambda r:r['offset'])
    after={r['offset']:ordered[i+1]['offset'] for i,r in enumerate(ordered[:-1])}
    blocks=[]
    for r in sorted(selected,key=lambda r:r['offset']):
        start=r['offset'];end=after.get(start,start+30+len(r['name'].encode('utf-8'))+r['compressed']+256)
        if blocks and start-blocks[-1]['end']<1024 and end-blocks[-1]['start']<=7*1024*1024:
            blocks[-1]['end']=end;blocks[-1]['rows'].append(r)
        else:blocks.append({'start':start,'end':end,'rows':[r]})
    print('Frozen raw candidates:',len(selected),'blocks',len(blocks),'download MiB',round(sum(b['end']-b['start'] for b in blocks)/1024**2,2),flush=True)
    def download(block):
        cached=all(Path(r['path']).exists() and binascii.crc32(Path(r['path']).read_bytes())&0xffffffff==r['crc'] for r in block['rows'])
        if not cached:
            for attempt in range(4):
                try:
                    request=urllib.request.Request(url,headers={'User-Agent':'TomatoCNN-research','Range':f"bytes={block['start']}-{block['end']-1}"})
                    with urllib.request.urlopen(request,timeout=120) as response:
                        assert response.status==206
                        data=response.read()
                    assert len(data)==block['end']-block['start']
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(attempt+1)
            for r in block['rows']:
                offset=r['offset']-block['start']
                signature,version,flags,method,mtime,mdate,crc,compressed,size,nlen,elen=struct.unpack_from('<4s5H3I2H',data,offset)
                assert signature==b'PK\x03\x04' and not flags&1
                name=data[offset+30:offset+30+nlen].decode('utf-8' if flags&0x800 else 'cp437')
                assert name==r['name']
                payload=data[offset+30+nlen+elen:offset+30+nlen+elen+r['compressed']]
                content=zlib.decompress(payload,-15) if method==8 else payload
                assert method in (0,8) and len(content)==r['size'] and binascii.crc32(content)&0xffffffff==r['crc']
                path=Path(r['path']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        return [{**r,'source':'Pakistan_matched_unseen','source_member':r['name'],'sha256':hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),
                 'source_url':protocol['source_url'],'reference_basis':'Explicit publisher raw-image category; no labels obtained from model predictions'} for r in block['rows']]
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download,block) for block in blocks]):
            rows.extend(future.result());print('Verified raw images',len(rows),'/830',flush=True)
    rows.sort(key=lambda r:r['source_member']);write_csv(DIRECTORY/'download_manifest.csv',rows)
    print('Completed raw candidate download',dict(Counter(r['category'] for r in rows)),flush=True)

if __name__=='__main__':main()
