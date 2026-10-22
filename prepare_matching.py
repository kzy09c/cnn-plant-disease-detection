"""Audit an unused source with disease categories matched to the training vocabulary."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib, json
import numpy as np
from PIL import Image, ImageOps
from prepare_data import ROOT, read_csv, write_csv, save_json, fingerprint, ssim, BKTree

DIRECTORY=ROOT/'data/3mbnb82mxd_unseen'

def main():
    assert not (ROOT/'evaluation_unseen_matching/metrics.json').exists(), 'Completed external test is frozen'
    incoming=read_csv(DIRECTORY/'download_manifest.csv')
    assert len(incoming)==830
    reference_path=ROOT/'data/field7/manifest.csv'
    configuration=json.loads((ROOT/'runs/mixed_attention_wild_v1/configuration.json').read_text())
    reference_hash=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert reference_hash==configuration['manifest_sha256']
    protocol=json.loads((DIRECTORY/'protocol_before_predictions.json').read_text())
    reference=read_csv(reference_path)
    training_categories={r['category'] for r in reference if r['source']=='PlantVillage' and r['split']=='train'}
    assert set(protocol['training_category_mapping'].values())<=training_categories, (protocol['training_category_mapping'], sorted(training_categories))
    reference+=read_csv(ROOT/'data/gtld_examples/manifest.csv')
    reference+=read_csv(ROOT/'data/agrivision4_unseen/manifest.csv')
    parents=set()
    for r in list(reference):
        if r.get('parent_path') and r['parent_path'] not in parents:
            parents.add(r['parent_path'])
            reference.append({'path':r['parent_path'],'phash':r['parent_phash'],'pixel_sha256':r['parent_pixel_sha256'],
                              'source':'TomatoDisease_parent','split':'reference_parent'})
    for r in reference:
        if not r.get('phash'):
            with Image.open(r['path']) as opened:
                r['phash']=f'{fingerprint(ImageOps.exif_transpose(opened).convert("RGB"))[0]:016x}'
    pixels=defaultdict(list);tree=BKTree()
    for i,r in enumerate(reference):
        pixels[r['pixel_sha256']].append(i);tree.add(int(r['phash'],16),i)
    reference_thumbs={}
    def load(r):
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
        with Image.open(r['path']) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            ph,thumb=fingerprint(image)
            pixel=hashlib.sha256(image.tobytes()).hexdigest()
            array=np.asarray(image.resize((160,160),Image.Resampling.BILINEAR)).transpose(2,0,1)
            width,height=image.size
        matches=[]
        for j in pixels.get(pixel,[]):
            matches.append({'reference_index':j,'kind':'exact decoded pixels','ssim':1.0})
        for j in tree.query(ph):
            if j not in reference_thumbs:
                with Image.open(reference[j]['path']) as opened:
                    reference_thumbs[j]=fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))[1]
            score=ssim(thumb,reference_thumbs[j])
            if score>=.97 and not any(m['reference_index']==j for m in matches):
                matches.append({'reference_index':j,'kind':'pHash<=4 and SSIM>=.97','ssim':score})
        r.update(pixel_sha256=pixel,phash=f'{ph:016x}',width=width,height=height,
                 overlap_reference_sources=';'.join(sorted({reference[m['reference_index']]['source'] for m in matches})),
                 overlap_reference_splits=';'.join(sorted({reference[m['reference_index']].get('split','') for m in matches})),
                 overlap_count=len(matches))
        return r,array,thumb,matches
    print('Auditing 830 raw images against',len(reference),'recorded images/parents',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded=list(pool.map(load,incoming))
    rows=[item[0] for item in loaded]
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    exact={};local=BKTree();pairs=[]
    binary_by_pixel=defaultdict(set)
    for i,r in enumerate(rows):
        binary_by_pixel[r['pixel_sha256']].add(r['label'])
        if r['pixel_sha256'] in exact:union(i,exact[r['pixel_sha256']])
        else:exact[r['pixel_sha256']]=i
        ph=int(r['phash'],16)
        for j in local.query(ph):
            score=ssim(loaded[i][2],loaded[j][2])
            if score>=.97:
                union(i,j);pairs.append({'first':i,'second':j,'ssim':score,'phash_distance':(ph^int(rows[j]['phash'],16)).bit_count()})
        local.add(ph,i)
    families=defaultdict(list)
    for i in range(len(rows)):families[find(i)].append(i)
    seen_pixels=set()
    for group,indices in families.items():
        reasons=[]
        if any(loaded[i][3] for i in indices):reasons.append('detected overlap family with prior data')
        if any(len(binary_by_pixel[rows[i]['pixel_sha256']])>1 for i in indices):reasons.append('contradictory identical-pixel health references')
        for i in indices:
            r=rows[i]
            exclusion='; '.join(reasons)
            if r['pixel_sha256'] in seen_pixels:
                exclusion=(exclusion+'; ' if exclusion else '')+'duplicate decoded photo within new source'
            seen_pixels.add(r['pixel_sha256'])
            r.update(index=i,group=f'matched_{group}',merged_group=f'matched_{group}',
                     split='excluded' if exclusion else 'external_test',exclusion=exclusion)
    write_csv(DIRECTORY/'manifest.csv',rows)
    np.save(DIRECTORY/'images.npy',np.stack([item[1] for item in loaded]))
    save_json(DIRECTORY/'local_similarity_pairs.json',pairs)
    save_json(DIRECTORY/'reference_overlap_matches.json',[
        {'candidate':i,'source_member':rows[i]['source_member'],
         'matches':[{**m,'source':reference[m['reference_index']]['source'],'path':reference[m['reference_index']]['path'],'split':reference[m['reference_index']].get('split','')} for m in item[3]]}
        for i,item in enumerate(loaded) if item[3]])
    eligible=[r for r in rows if not r['exclusion']]
    audit={'source_url':'https://data.mendeley.com/datasets/3mbnb82mxd/2',
           'downloaded_originals':len(rows),'downloaded_classes':dict(Counter(r['binary_class'] for r in rows)),
           'eligible_images':len(eligible),'eligible_classes':dict(Counter(r['binary_class'] for r in eligible)),
           'eligible_categories':dict(Counter(r['category'] for r in eligible)),
           'eligible_detected_groups':len({r['merged_group'] for r in eligible}),
           'exclusions':dict(Counter(r['exclusion'] for r in rows if r['exclusion'])),
           'reference_records_including_parents':len(reference),'reference_manifest_sha256':reference_hash,
           'new_manifest_sha256':hashlib.sha256((DIRECTORY/'manifest.csv').read_bytes()).hexdigest(),
           'image_cache_sha256':hashlib.sha256((DIRECTORY/'images.npy').read_bytes()).hexdigest(),
           'duplicate_policy':'Exclude entire detected families overlapping any prior audited data; exclude contradictory identical-pixel families; retain one per decoded pixel hash; group pHash<=4 + SSIM>=.97 near matches for uncertainty.',
           'label_policy':'Publisher original-image categories retained. No model inference has occurred during selection/audit.',
           'limitations':'Publisher labels are not independently diagnosed here. Similarity checks may miss crops, alternate views and garden/session relationships. Group intervals do not capture those missing relationships.'}
    save_json(DIRECTORY/'audit_before_predictions.json',audit)
    # Fixed source-stratified previews for scope inspection, never selected by model scores.
    canvas=Image.new('RGB',(5*180,len({r['category'] for r in rows})*205),'white')
    from PIL import ImageDraw
    draw=ImageDraw.Draw(canvas)
    rng=np.random.default_rng(20261022)
    for line,category in enumerate(sorted({r['category'] for r in rows})):
        subset=[r for r in rows if r['category']==category]
        sample=[subset[j] for j in rng.choice(len(subset),5,replace=False)]
        for col,r in enumerate(sample):
            with Image.open(r['path']) as opened:
                image=ImageOps.exif_transpose(opened).convert('RGB');image.thumbnail((180,175))
                canvas.paste(image,(col*180,line*205+25))
            draw.text((col*180+3,line*205+3),category[:23],fill='black')
            draw.text((col*180+3,line*205+185),f"candidate {r['index']}",fill='black')
    canvas.save(DIRECTORY/'preprediction_scope_preview.jpg')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
