"""Audit annotated leaf regions; add only non-overlapping, labelled examples."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageOps
from prepare_data import ROOT,DATA,read_csv,write_csv,save_json,BKTree,fingerprint,ssim

def canonical(name):
    name=Path(name).stem.split('.rf.')[0]
    name=re.sub(r'_(jpg|jpeg|png)$','',name,flags=re.I)
    return re.sub('[^a-z0-9]','',name.lower())

def main():
    source=DATA/'tomato_disease/extracted/Tomato-Disease'
    if __import__('os').name=='nt':source=Path('\\\\?\\'+str(source.resolve()))
    labels=sorted(source.rglob('labels/*.txt'))
    counts=Counter()
    for path in labels:
        for line in path.read_text().splitlines():
            if line.strip():counts[int(line.split()[0])]+=1
    # The release lacks a YAML class map. Recover the binary Healthy identity
    # using the uniquely matching published annotation total, then review examples.
    paper=ET.parse(DATA/'sources/tdd_paper.xml')
    table=next(e for e in paper.iter('table-wrap') if e.attrib.get('id','').endswith('002'))
    published={}
    for row in table.iter('tr'):
        cells=[''.join(c.itertext()).strip() for c in row if c.tag in {'td','th'}]
        if cells and cells[0]=='Healthy':published['healthy_total']=int(cells[-1])
    assert published['healthy_total']==736
    healthy_ids=[i for i,n in counts.items() if n==published['healthy_total']]
    assert len(healthy_ids)==1,'Healthy numerical class must match the primary-source annotation count unambiguously'
    healthy_id=healthy_ids[0]
    save_json(DATA/'tomato_disease/binary_label_mapping.json',{'healthy_id':healthy_id,'observed_annotation_counts':dict(counts),
              'basis':'Inference from unique annotation total matching Healthy=736 in the releasing authors Table 2; not an official supplied YAML. Class-wise examples reviewed before training.',
              'primary_source':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334989',
              'paper_sha256':hashlib.sha256((DATA/'sources/tdd_paper.xml').read_bytes()).hexdigest(),
              'published_annotation_total':6585,'actual_annotation_total':sum(counts.values())})
    print('Numerical Healthy ID',healthy_id,'counts',counts,flush=True)
    base=read_csv(DATA/'field/manifest.csv')
    legacy=read_csv(DATA/'manifest.csv')
    reference_names=set()
    for row in base:
        for key in ['path','source_path','source_member']:
            name=canonical(row.get(key,''))
            if len(name)>=16 and not re.fullmatch(r'img\d+',name):reference_names.add(name)
    reference_pixels={r['pixel_sha256'] for r in base}
    reference_thumbs=list(np.load(DATA/'thumbnails.npy'))
    reference_hashes=[int(r['phash'],16) for r in legacy]
    # Tomato-Village full-photo fingerprints were separately checked against legacy data.
    tv=read_csv(DATA/'external_manifest.csv')
    for r in tv:
        with Image.open(r['path']) as opened:
            ph,thumb=fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))
        reference_hashes.append(ph);reference_thumbs.append(thumb)
    reference_tree=BKTree()
    for i,ph in enumerate(reference_hashes):reference_tree.add(ph,i)
    crop_dir=DATA/'tomato_disease/crops'
    crop_dir.mkdir(exist_ok=True)
    def load(label):
        image_path=label.parent.parent/'images'/(label.stem+'.jpg')
        assert image_path.exists(),image_path
        family=canonical(image_path.name)
        file_sha=hashlib.sha256(image_path.read_bytes()).hexdigest()
        annotations=[]
        for line in label.read_text().splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()))
            assert len(values)==5 and values[0].is_integer()
            cls,cx,cy,w,h=values
            assert int(cls) in counts and 0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1
            annotations.append((int(cls),cx,cy,w,h))
        with Image.open(image_path) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            full_pixel=hashlib.sha256(image.tobytes()).hexdigest()
            full_ph,full_thumb=fingerprint(image)
            exclusion='filename identity overlaps earlier sources' if len(family)>=16 and family in reference_names else ''
            if re.search(r'(?:^|[_\W])(?:potato|pepper|apple|grape|cucumber|eggplant|squash|wheat|rice)s?(?:[_\W]|$)',image_path.name,re.I):
                exclusion='source filename explicitly identifies another crop'
            if full_pixel in reference_pixels:exclusion='decoded photo overlaps earlier sources'
            for j in reference_tree.query(full_ph):
                if ssim(full_thumb,reference_thumbs[j])>=.97:exclusion='structural photo overlap with earlier sources';break
            crops=[]
            skipped=0
            for a,(cls,cx,cy,w,h) in enumerate(annotations):
                if min(w*image.width,h*image.height)<32:skipped+=1;continue
                # Fixed 10% context margin, specified before evaluation.
                x0=max(0,int((cx-w*.55)*image.width));x1=min(image.width,int(np.ceil((cx+w*.55)*image.width)))
                y0=max(0,int((cy-h*.55)*image.height));y1=min(image.height,int(np.ceil((cy+h*.55)*image.height)))
                crop=image.crop((x0,y0,x1,y1))
                ph,thumb=fingerprint(crop)
                pixels=hashlib.sha256(crop.tobytes()).hexdigest()
                reason=exclusion
                if pixels in reference_pixels:reason='decoded crop overlaps earlier sources'
                for j in reference_tree.query(ph):
                    if ssim(thumb,reference_thumbs[j])>=.97:reason='structural crop overlap with earlier sources';break
                target=crop_dir/(hashlib.sha1(str(image_path).encode()).hexdigest()+f'_{a}.png')
                crop.save(target)
                row={'source':'TomatoDisease_ROI','path':str(target.resolve()),'parent_path':str(image_path.resolve()),
                     'source_member':str(image_path.relative_to(source)),'parent_sha256':file_sha,'parent_pixel_sha256':full_pixel,
                     'parent_phash':f'{full_ph:016x}','family':family,'publisher_split':image_path.relative_to(source).parts[0],
                     'group':family,'category':f'annotation_{cls}','annotation_class':cls,'label':int(cls!=healthy_id),
                     'binary_class':'Healthy' if cls==healthy_id else 'Unhealthy','pixel_sha256':pixels,'phash':f'{ph:016x}',
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bbox_pixels':json.dumps([x0,y0,x1,y1]),'exclusion':reason}
                small=np.asarray(crop.resize((160,160),Image.Resampling.BILINEAR)).transpose(2,0,1)
                crops.append((row,small,thumb))
        return crops,full_ph,full_thumb,family,full_pixel,skipped,str(image_path)
    # Ordered results make indices and manifests reproducible.
    with ThreadPoolExecutor(max_workers=8) as pool:loaded=list(pool.map(load,labels))
    rows=[];arrays=[];thumbs=[];parents=[]
    for crops,ph,thumb,family,pixel,skipped,path in loaded:
        start=len(rows)
        for row,small,th in crops:rows.append(row);arrays.append(small);thumbs.append(th)
        parents.append({'start':start,'end':len(rows),'ph':ph,'thumb':thumb,'family':family,'pixel':pixel,'path':path,'small_boxes_skipped':skipped})
    del loaded
    n=len(rows);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    seen={};tree=BKTree()
    for i,r in enumerate(rows):
        for key in [('family',r['family']),('pixel',r['pixel_sha256']),('parent',r['parent_pixel_sha256'])]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        ph=int(r['phash'],16)
        for j in tree.query(ph):
            if find(i)!=find(j) and ssim(thumbs[i],thumbs[j])>=.97:union(i,j)
        tree.add(ph,i)
    p_tree=BKTree();p_lookup=[]
    for item in parents:
        if item['start']==item['end']:continue
        for j in p_tree.query(item['ph']):
            prior=p_lookup[j]
            if ssim(item['thumb'],prior['thumb'])>=.97:union(item['start'],prior['start'])
        p_tree.add(item['ph'],len(p_lookup));p_lookup.append(item)
    groups=defaultdict(list)
    for i in range(n):groups[find(i)].append(i)
    exact=defaultdict(set)
    for r in rows:exact[r['pixel_sha256']].add(r['label'])
    conflicts={p for p,labels_ in exact.items() if len(labels_)>1}
    priority={'train':0,'val':1,'valid':1,'test':2};names={'train':'train','val':'validation','valid':'validation','test':'test'}
    for g,indices in groups.items():
        split=names[max((rows[i]['publisher_split'] for i in indices),key=priority.get)]
        reasons={rows[i]['exclusion'] for i in indices if rows[i]['exclusion']}
        if any(rows[i]['pixel_sha256'] in conflicts for i in indices):reasons.add('identical crop has contradictory labels')
        for i in indices:
            rows[i]['merged_group']='td_'+str(g)
            rows[i]['split']='excluded' if reasons else split
            rows[i]['exclusion']='; '.join(sorted(reasons))
            rows[i]['index']=len(base)+i
    out=DATA/'field6';out.mkdir(exist_ok=True)
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype=np.uint8,shape=(len(base)+n,3,160,160))
    original=np.load(DATA/'field/images.npy',mmap_mode='r')
    for start in range(0,len(base),1000):
        end=min(start+1000,len(base));images[start:end]=original[start:end]
    for i,small in enumerate(arrays):images[len(base)+i]=small
    images.flush()
    write_csv(out/'manifest.csv',base+rows)
    counts_final=Counter((r['split'],r['binary_class']) for r in rows)
    audit={'new_source':'https://zenodo.org/records/15868289','counts':{' / '.join(k):v for k,v in counts_final.items()},
           'original_photos':len(labels),'crop_examples':n,'groups':len(groups),'small_annotations_skipped':sum(r['small_boxes_skipped'] for r in parents),
           'exclusions':dict(Counter(r['exclusion'] for r in rows if r['exclusion'])),'healthy_annotation_id':healthy_id,
           'class_mapping_status':'Inferred from unique match to the authors Healthy annotation total; visual review required before training',
           'evaluation_unit':'Annotated regions with 10% context margin, not automatic localization or whole-scene classification',
           'grouping':'Canonical pre-Roboflow photo name, exact photo/crop pixels, pHash<=4 plus SSIM>=.97 for photos and crops; highest-reserved publisher partition wins',
           'manifest_sha256':hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest(),
           'limitations':'Field and online source mixture. Complete web provenance/plant/session IDs absent. Class mapping was recovered, not supplied by an official YAML; annotation-count agreement does not independently validate biology.'}
    save_json(out/'audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
