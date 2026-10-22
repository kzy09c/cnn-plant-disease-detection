"""Conservatively detect resized/cropped photo copies before CNN inference."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib, json, sys
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from prepare_data import ROOT, read_csv, write_csv, save_json

sys.path.insert(0,str(ROOT/'_publish/audit_deps'))
import cv2
cv2.setNumThreads(1)
DIRECTORY=ROOT/'data/3mbnb82mxd_unseen'

@lru_cache(maxsize=None)
def features(path):
    with Image.open(path) as opened:
        image=ImageOps.exif_transpose(opened).convert('L')
        image.thumbnail((512,512),Image.Resampling.LANCZOS)
        gray=np.asarray(image).copy()
    keypoints,descriptors=cv2.SIFT_create(nfeatures=500).detectAndCompute(gray,None)
    points=np.asarray([k.pt for k in keypoints],dtype=np.float32)
    return gray,points,descriptors

def geometric_match(first,second):
    a,pa,da=features(first);b,pb,db=features(second)
    if da is None or db is None or min(len(da),len(db))<20:return None
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.7*p[1].distance]
    if len(good)<20:return None
    p=np.float32([pa[m.queryIdx] for m in good]);q=np.float32([pb[m.trainIdx] for m in good])
    h,mask=cv2.findHomography(p,q,cv2.RANSAC,3.,maxIters=1500,confidence=.995)
    if h is None or mask is None or not np.isfinite(h).all():return None
    inliers=mask.ravel().astype(bool);n=int(inliers.sum())
    if n<20 or n/len(good)<.7:return None
    area_a=cv2.contourArea(cv2.convexHull(p[inliers]))/(a.shape[0]*a.shape[1])
    area_b=cv2.contourArea(cv2.convexHull(q[inliers]))/(b.shape[0]*b.shape[1])
    if min(area_a,area_b)<.04:return None
    warped=cv2.warpPerspective(a,h,(b.shape[1],b.shape[0]))
    valid=cv2.warpPerspective(np.ones_like(a,dtype=np.uint8),h,(b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)
    valid=cv2.erode(valid,np.ones((5,5),np.uint8)).astype(bool)
    if valid.sum()<1000:return None
    x=warped[valid].astype(float);y=b[valid].astype(float)
    if min(x.std(),y.std())<5:return None
    correlation=float(np.corrcoef(x,y)[0,1])
    if correlation<.90:return None
    return {'inliers':n,'good_matches':len(good),'inlier_ratio':n/len(good),'matched_area_fractions':[area_a,area_b],
            'aligned_gray_correlation':correlation,'homography':h.tolist()}

def main():
    assert not (ROOT/'evaluation_unseen_matching/metrics.json').exists(), 'No cohort changes after inference'
    rows=read_csv(DIRECTORY/'manifest.csv')
    audit=json.loads((DIRECTORY/'audit_before_predictions.json').read_text())
    assert len(rows)==830 and audit['eligible_images']==498
    (DIRECTORY/'manifest_before_geometric_audit.csv').write_bytes((DIRECTORY/'manifest.csv').read_bytes())
    save_json(DIRECTORY/'audit_before_geometry.json',audit)
    protocol=json.loads((DIRECTORY/'protocol_before_predictions.json').read_text())
    method={'purpose':'Classical, nonlearned duplicate detection only; not CNN features or training.',
            'opencv':cv2.__version__,'features':'SIFT, max 500 features, longest edge 512',
            'shortlists':'For each candidate: top 20 other new-image pHashes within Hamming 26; then top 12 distinct prior-image pHashes within Hamming 26 for retained family representatives.',
            'confirmation':'L2 Lowe ratio .7; at least 20 homography RANSAC inliers (3px) and >=.7 inlier ratio; convex hull >=.04 area in both; aligned grayscale correlation >=.90 over >=1000 pixels.',
            'policy':'Propagate prior overlap across confirmed photo families. Exclude contradictory binary labels in a family. Keep one representative per remaining photo family. Frozen before any CNN predictions.',
            'limitation':'Shortlisting is incomplete and geometric checks can miss heavily transformed copies or different views. Does not prove garden/session independence.'}
    protocol['additional_geometric_audit_before_predictions']=method
    save_json(DIRECTORY/'protocol_before_predictions.json',protocol)
    hashes=np.asarray([int(r['phash'],16) for r in rows],dtype=np.uint64)
    pairs=set()
    for i,h in enumerate(hashes):
        distance=np.bitwise_count(hashes^h);distance[i]=255
        for j in np.argsort(distance)[:20]:
            if distance[j]<=26:pairs.add(tuple(sorted((i,int(j)))))
    print('Extracting classical features for 830 images; local candidate pairs',len(pairs),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(features,[r['path'] for r in rows]))
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    initial={}
    for i,r in enumerate(rows):
        if r['merged_group'] in initial:union(i,initial[r['merged_group']])
        else:initial[r['merged_group']]=i
    def check_pair(pair):
        i,j=pair;match=geometric_match(rows[i]['path'],rows[j]['path'])
        return {'first':i,'second':j,**match} if match else None
    local_matches=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for number,result in enumerate(pool.map(check_pair,sorted(pairs)),1):
            if result:local_matches.append(result);union(result['first'],result['second'])
            if number%2000==0:print('Local geometry pairs checked',number,'confirmed',len(local_matches),flush=True)
    families=defaultdict(list)
    for i in range(len(rows)):families[find(i)].append(i)
    known_roots={find(i) for i,r in enumerate(rows) if 'overlap family with prior data' in r['exclusion']}
    candidates=[indices[0] for group,indices in families.items() if group not in known_roots and not all(rows[i]['exclusion'] for i in indices)]
    print('Local photo families',len(families),'remaining representatives for prior check',len(candidates),flush=True)
    reference=read_csv(ROOT/'data/field7/manifest.csv')+read_csv(ROOT/'data/agrivision4_unseen/manifest.csv')
    parents=set()
    for r in list(reference):
        if r.get('parent_path') and r['parent_path'] not in parents:
            parents.add(r['parent_path']);reference.append({'path':r['parent_path'],'phash':r['parent_phash'],
                'pixel_sha256':r['parent_pixel_sha256'],'source':'TomatoDisease_parent','split':'reference_parent'})
    unique={}
    for r in reference:
        if r.get('phash'):unique.setdefault(r['pixel_sha256'],r)
    reference=list(unique.values())
    rh=np.asarray([int(r['phash'],16) for r in reference],dtype=np.uint64)
    def check_prior(i):
        distances=np.bitwise_count(rh^hashes[i])
        for j in np.argsort(distances)[:12]:
            if distances[j]>26:continue
            r=reference[int(j)];match=geometric_match(rows[i]['path'],r['path'])
            if match:return {'candidate':i,'reference':{k:r.get(k,'') for k in ['source','split','path','phash','pixel_sha256']},**match}
        return None
    prior_matches=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for number,result in enumerate(pool.map(check_prior,candidates),1):
            if result:prior_matches.append(result);known_roots.add(find(result['candidate']))
            if number%100==0:print('Prior family representatives checked',number,'new overlaps',len(prior_matches),flush=True)
    for group,indices in families.items():
        labels={rows[i]['label'] for i in indices}
        already_kept=False
        for i in indices:
            r=rows[i];reasons=[]
            if group in known_roots:reasons.append('photo family overlaps earlier data (hash/structural or geometric)')
            if len(labels)>1:reasons.append('confirmed photo family has conflicting binary references')
            if not reasons and already_kept:reasons.append('duplicate/cropped/resized photo within new source')
            if not reasons:already_kept=True
            r.update(exclusion='; '.join(reasons),split='excluded' if reasons else 'external_test',
                     group=f'geometry_{group}',merged_group=f'geometry_{group}')
    write_csv(DIRECTORY/'manifest.csv',rows)
    eligible=[r for r in rows if not r['exclusion']]
    audit.update(eligible_images=len(eligible),eligible_classes=dict(Counter(r['binary_class'] for r in eligible)),
                 eligible_categories=dict(Counter(r['category'] for r in eligible)),eligible_detected_groups=len({r['group'] for r in eligible}),
                 exclusions=dict(Counter(r['exclusion'] for r in rows if r['exclusion'])),
                 new_manifest_sha256=hashlib.sha256((DIRECTORY/'manifest.csv').read_bytes()).hexdigest(),
                 additional_geometric_audit=method,confirmed_local_geometric_pairs=len(local_matches),
                 additional_prior_geometric_families=len(prior_matches),prior_geometric_distinct_reference_images=len(reference))
    save_json(DIRECTORY/'audit_before_predictions.json',audit)
    save_json(DIRECTORY/'geometric_local_matches.json',local_matches)
    save_json(DIRECTORY/'geometric_prior_matches.json',prior_matches)
    # QA examples with the weakest accepted geometric confirmations.
    examples=[(rows[m['first']]['path'],rows[m['second']]['path'],m) for m in sorted(local_matches,key=lambda m:m['inliers'])[:4]]
    examples += [(rows[m['candidate']]['path'],m['reference']['path'],m) for m in sorted(prior_matches,key=lambda m:m['inliers'])[:4]]
    canvas=Image.new('RGB',(600,220*len(examples)),'white');draw=ImageDraw.Draw(canvas)
    for line,(a,b,m) in enumerate(examples):
        for col,path in enumerate([a,b]):
            with Image.open(path) as opened:
                image=ImageOps.exif_transpose(opened).convert('RGB');image.thumbnail((295,190));canvas.paste(image,(col*300,line*220+25))
        draw.text((5,line*220+3),f"inliers={m['inliers']}, aligned correlation={m['aligned_gray_correlation']:.3f}",fill='black')
    if examples:canvas.save(DIRECTORY/'geometric_match_qa.jpg')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
