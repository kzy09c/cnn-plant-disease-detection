"""Freeze the seven published GTLD example photos as a small unseen sanity check."""
from pathlib import Path
import json,hashlib,urllib.request
import numpy as np
from PIL import Image,ImageOps
from prepare_data import ROOT,DATA,read_csv,write_csv,save_json,fingerprint,ssim,BKTree

if __name__=='__main__':
    source=json.loads((DATA/'sources/2bdfjb99k5.json').read_text())
    out=DATA/'gtld_examples';out.mkdir(exist_ok=True)
    rows=[];arrays=[]
    reference=read_csv(DATA/'field6/manifest.csv')
    pixels={r['pixel_sha256'] for r in reference}
    tree=BKTree()
    for i,r in enumerate(reference):tree.add(int(r['phash'],16),i)
    thumbs={}
    for file in source['files']:
        path=out/file['filename']
        if not path.exists():
            request=urllib.request.Request(file['content_details']['download_url'],headers={'User-Agent':'TomatoCNN-research'})
            with urllib.request.urlopen(request,timeout=30) as response:path.write_bytes(response.read())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==file['content_details']['sha256_hash']
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB');ph,thumb=fingerprint(image)
            pixel=hashlib.sha256(image.tobytes()).hexdigest()
            small=np.asarray(image.resize((160,160),Image.Resampling.BILINEAR)).transpose(2,0,1)
        exclusion='exact overlap' if pixel in pixels else ''
        for j in tree.query(ph):
            if j not in thumbs:
                with Image.open(reference[j]['path']) as opened:thumbs[j]=fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))[1]
            if ssim(thumb,thumbs[j])>=.97:exclusion='structural overlap';break
        healthy='healthy' in file['filename'].lower()
        rows.append({'index':len(rows),'source':'GTLD_examples','path':str(path.resolve()),'sha256':file['content_details']['sha256_hash'],
                     'pixel_sha256':pixel,'binary_class':'Healthy' if healthy else 'Unhealthy','label':int(not healthy),
                     'category':path.stem,'group':path.stem,'exclusion':exclusion,'reference_basis':'Explicit publisher filename; CC BY 4.0','split':'external_test'})
        arrays.append(small)
    write_csv(out/'manifest.csv',rows);np.save(out/'images.npy',np.stack(arrays))
    save_json(out/'audit.json',{'source_url':'https://data.mendeley.com/datasets/2bdfjb99k5/1','n':len(rows),
                               'excluded':sum(bool(r['exclusion']) for r in rows),'role':'Unseen illustrative sanity check, only one healthy photo. Not a substantial test dataset or representative internet sample.'})
    print('Frozen GTLD example images',len(rows),'overlaps',sum(bool(r['exclusion']) for r in rows),flush=True)
