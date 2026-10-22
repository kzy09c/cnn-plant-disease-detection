"""Add audited tomato-only internet scenes with source-page and duplicate groups."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
import urllib.parse
import numpy as np
from PIL import Image,ImageOps
from prepare_data import ROOT,DATA,read_csv,write_csv,save_json,BKTree,fingerprint,ssim

def page_key(url):
    if not url:return ''
    parsed=urllib.parse.urlsplit(url)
    query=urllib.parse.urlencode([(k,v) for k,v in urllib.parse.parse_qsl(parsed.query) if not k.lower().startswith('utm_') and k.lower() not in ['fbclid','gclid']])
    return urllib.parse.urlunsplit(('',parsed.netloc.lower(),parsed.path.rstrip('/'),query,''))

def main():
    directory=DATA/'plantwild'
    incoming=read_csv(directory/'download_manifest.csv')
    review={r['source_member']:r for r in read_csv(directory/'healthy_review.csv')}
    base=read_csv(DATA/'field6/manifest.csv')
    reference=base+read_csv(DATA/'gtld_examples/manifest.csv')
    # Include full parent photos of every annotated region, not only their crops.
    seen_parents=set()
    for r in base:
        if r.get('parent_path') and r['parent_path'] not in seen_parents:
            seen_parents.add(r['parent_path'])
            reference.append({'path':r['parent_path'],'phash':r['parent_phash'],'pixel_sha256':r['parent_pixel_sha256']})
    tree=BKTree()
    pixels={r['pixel_sha256'] for r in reference}
    urls={page_key(r.get('source_url','')) for r in reference if r.get('source_url')}
    for i,r in enumerate(reference):
        # GTLD's seven examples are independently fingerprinted here.
        if not r.get('phash'):
            with Image.open(r['path']) as opened:r['phash']=f"{fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))[0]:016x}"
        tree.add(int(r['phash'],16),i)
    reference_thumbs={}
    def load(row):
        path=Path(row['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            ph,thumb=fingerprint(image)
            pixel=hashlib.sha256(image.tobytes()).hexdigest()
            small=np.asarray(image.resize((160,160),Image.Resampling.BILINEAR)).transpose(2,0,1)
        overlap='decoded overlap with earlier photos' if pixel in pixels else ''
        key=page_key(row['source_url'])
        if key and key in urls:overlap='same source page as earlier photos'
        for j in tree.query(ph):
            if j not in reference_thumbs:
                with Image.open(reference[j]['path']) as opened:
                    reference_thumbs[j]=fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))[1]
            if ssim(thumb,reference_thumbs[j])>=.97:overlap='structural overlap with earlier photos';break
        row.update(pixel_sha256=pixel,phash=f'{ph:016x}',page_group=key,overlap_exclusion=overlap,
                   review_exclusion=review[row['source_member']]['review_exclusion'] if row['binary_class']=='Healthy' else '',
                   label_basis='Published tomato disease category; Healthy generic leaf class visually screened for apparent absence of symptoms')
        return row,small,thumb
    with ThreadPoolExecutor(max_workers=8) as pool:loaded=list(pool.map(load,incoming))
    rows=[x[0] for x in loaded];arrays=[x[1] for x in loaded];thumbs=[x[2] for x in loaded]
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    seen={};local=BKTree()
    for i,row in enumerate(rows):
        keys=[('pixel',row['pixel_sha256'])]
        if row['page_group']:keys.append(('page',row['page_group']))
        for key in keys:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        ph=int(row['phash'],16)
        for j in local.query(ph):
            if find(i)!=find(j) and ssim(thumbs[i],thumbs[j])>=.97:union(i,j)
        local.add(ph,i)
    exact=defaultdict(set)
    for r in rows:exact[r['pixel_sha256']].add(r['label'])
    conflicts={p for p,labels in exact.items() if len(labels)>1}
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    priority={'train':0,'validation':1,'test':2}
    for group,indices in groups.items():
        split=max((rows[i]['publisher_split'] for i in indices),key=priority.get)
        reasons={rows[i]['overlap_exclusion'] for i in indices if rows[i]['overlap_exclusion']}
        if any(rows[i]['pixel_sha256'] in conflicts for i in indices):reasons.add('identical pixels have conflicting binary references')
        for i in indices:
            row=rows[i];reason='; '.join(sorted(reasons|({row['review_exclusion']} if row['review_exclusion'] else set())))
            row.update(split='excluded' if reason else split,exclusion=reason,merged_group='pw_'+str(group),group='pw_'+str(group),index=len(base)+i)
    out=DATA/'field7';out.mkdir(exist_ok=True)
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype=np.uint8,shape=(len(base)+len(rows),3,160,160))
    original=np.load(DATA/'field6/images.npy',mmap_mode='r')
    for start in range(0,len(base),1000):
        end=min(start+1000,len(base));images[start:end]=original[start:end]
    for i,small in enumerate(arrays):images[len(base)+i]=small
    images.flush();write_csv(out/'manifest.csv',base+rows)
    counts=Counter((r['split'],r['binary_class']) for r in rows)
    audit={'source':'https://tqwei05.github.io/PlantWild/','release':'PlantWild v1, official uqtwei2 archive, tomato classes only',
           'counts':{' / '.join(k):v for k,v in sorted(counts.items())},'groups':len(groups),
           'exclusions':dict(Counter(r['exclusion'] for r in rows if r['exclusion'])),
           'grouping':'Normalized original source page, decoded duplicates, pHash<=4 plus SSIM>=.97; entire groups promoted to test > validation > train',
           'test_status':'New benchmark in this project; source chosen after earlier experimental results. No PlantWild predictions used in eligibility, splitting or model choice.',
           'healthy_reference':'Apparently asymptomatic images retained after manual pre-prediction screening of generic tomato leaf class; not independent diagnoses',
           'license':'CC BY-NC-ND 4.0; raw originals kept unmodified locally and not redistributed',
           'limitations':'Images sourced from internet search engines; original expert/plant/session identity incomplete, source categories may have errors. Conditional curated benchmark, not a probability sample of the internet.',
           'manifest_sha256':hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest()}
    save_json(out/'audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
