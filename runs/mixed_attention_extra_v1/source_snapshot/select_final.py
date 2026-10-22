"""Choose the delivered checkpoint using identical development validation only."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from model import LeafCNN
from train import probabilities,metrics,selection_score
from prepare_data import ROOT,read_csv,save_json

if __name__=='__main__':
    rows=read_csv(ROOT/'data/field6/manifest.csv')
    images=torch.from_numpy(np.load(ROOT/'data/field6/images.npy')).cuda()
    sources=sorted({r['source'] for r in rows if r['split']=='validation'})
    candidates=[]
    for selection_file in ['selection_field.json','selection_extra.json']:
        candidate=json.loads((ROOT/'runs'/selection_file).read_text())['selected']
        path=Path(candidate['checkpoint'])
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=LeafCNN(**checkpoint['architecture']).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint['state_dict'])
        results={}
        for source in sources:
            idx=np.asarray([i for i,r in enumerate(rows) if r['source']==source and r['split']=='validation'])
            y=np.asarray([int(rows[i]['label']) for i in idx])
            results[source]=metrics(y,probabilities(model,images,idx))
        candidate={**candidate,'common_validation_score':selection_score(results),'common_validation':results,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        candidates.append(candidate)
        print(candidate['name'],candidate['common_validation_score'],flush=True)
        del model
    winner=max(candidates,key=lambda x:x['common_validation_score'])
    save_json(ROOT/'runs/selection_final.json',{'candidates':candidates,'selected':winner,'basis':'Same six development-validation cohorts, fixed threshold .5, no test predictions used'})
    print('Selected final model using validation:',winner['name'],flush=True)
