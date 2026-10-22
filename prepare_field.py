"""Second, prospectively specified experiment adding grouped field photos."""
from collections import defaultdict,Counter
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image,ImageOps
from concurrent.futures import ThreadPoolExecutor
from prepare_data import ROOT,DATA,read_csv,write_csv,save_json,BKTree,ssim,fingerprint

def main():
    out=DATA/'field'
    out.mkdir(exist_ok=True)
    base=read_csv(DATA/'manifest.csv')
    field=read_csv(DATA/'external_manifest.csv')
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype=np.uint8,shape=(len(base)+len(field),3,160,160))
    for start in range(0,len(base),2000):
        end=min(start+2000,len(base))
        images[start:end]=np.load(DATA/'images.npy',mmap_mode='r')[start:end]
    images[len(base):]=np.load(DATA/'external_images.npy',mmap_mode='r')
    images.flush()
    def thumb(row):
        with Image.open(row['path']) as opened:
            return fingerprint(ImageOps.exif_transpose(opened).convert('RGB'))[1]
    with ThreadPoolExecutor(max_workers=8) as pool: thumbs=list(pool.map(thumb,field))
    parent=list(range(len(field)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    families={}
    exact={}
    tree=BKTree()
    pairs=[]
    for i,row in enumerate(field):
        for key in [('family',row['group']),('pixel',row['pixel_sha256'])]:
            if key in families:union(i,families[key])
            else:families[key]=i
        exact.setdefault(row['pixel_sha256'],set()).add(row['label'])
        ph=int(row['phash'],16)
        for j in tree.query(ph):
            if find(i)!=find(j) and ssim(thumbs[i],thumbs[j])>=.97:
                union(i,j)
                pairs.append([i,j])
        tree.add(ph,i)
    conflicts={p for p,labels in exact.items() if len(labels)>1}
    groups=defaultdict(list)
    for i in range(len(field)):groups[find(i)].append(i)
    order={'train':0,'val':1,'test':2}
    names={'train':'train','val':'validation','test':'test'}
    for group,indices in groups.items():
        split=names[max((field[i]['publisher_split'] for i in indices),key=order.get)]
        for i in indices:
            row=field[i]
            row['split']='excluded' if row['exclusion'] or any(field[j]['pixel_sha256'] in conflicts for j in indices) else split
            row['original_split']=row['publisher_split']
            row['merged_group']='tv_'+str(group)
            row['index']=len(base)+i
    write_csv(out/'manifest.csv',base+field)
    counts=Counter((r['source'],r['split'],r['binary_class']) for r in base+field)
    save_json(out/'audit.json',{'counts':{' / '.join(k):v for k,v in sorted(counts.items())},
                              'field_groups':len(groups),'additional_similarity_pairs':pairs,
                              'split_rule':'Preserve publisher partitions and promote entire inferred photo/duplicate families to test > validation > train',
                              'prospective_decision':'Field-source training planned from validation limitations before any test prediction in these experiments',
                              'manifest_sha256':hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest(),
                              'limitations':'No original plant/garden IDs. Adjacent captures can remain related despite photo grouping. Tests are exploratory benchmarks; not internet-wide validation.'})
    print({k:v for k,v in counts.items() if k[0]=='TomatoVillage'},flush=True)

if __name__=='__main__':main()
