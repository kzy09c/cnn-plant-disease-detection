"""Freeze the delivered model before examining the new PlantWild holdout."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from model import LeafCNN
from train import probabilities,metrics,selection_score
from prepare_data import ROOT,read_csv,save_json

if __name__=='__main__':
    rows=read_csv(ROOT/'data/field7/manifest.csv')
    images=torch.from_numpy(np.load(ROOT/'data/field7/images.npy')).cuda()
    candidates=[]
    sources=sorted({r['source'] for r in rows if r['split']=='validation'})
    for selection_file in ['selection_final.json','selection_wild_run.json']:
        candidate=json.loads((ROOT/'runs'/selection_file).read_text())['selected']
        path=Path(candidate['checkpoint']);checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=LeafCNN(**checkpoint['architecture']).cuda().to(memory_format=torch.channels_last);model.load_state_dict(checkpoint['state_dict'])
        results={}
        for source in sources:
            ids=np.asarray([i for i,r in enumerate(rows) if r['source']==source and r['split']=='validation'])
            results[source]=metrics(np.asarray([int(rows[i]['label']) for i in ids]),probabilities(model,images,ids))
        candidate={**candidate,'common_validation_score':selection_score(results),'common_validation':results,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        candidates.append(candidate);print(candidate['name'],candidate['common_validation_score'],flush=True);del model
    selected=max(candidates,key=lambda x:x['common_validation_score'])
    save_json(ROOT/'runs/selection_wild.json',{'candidates':candidates,'selected':selected,'basis':'Identical seven-source validation cohorts only; frozen before new PlantWild test predictions'})
    print('Frozen model',selected['name'],flush=True)
