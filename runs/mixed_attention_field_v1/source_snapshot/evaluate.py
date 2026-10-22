"""Freeze a validation-selected model, then evaluate without further tuning."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from model import LeafCNN
from train import metrics, probabilities
from prepare_data import ROOT, read_csv, write_csv, save_json

def bootstrap(rows, y, p, draws=1000):
    """Resample recorded groups; this does not account for missing farm IDs."""
    groups=np.asarray([r.get('merged_group',r.get('group','')) for r in rows])
    unique=np.unique(groups)
    members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(20261022)
    values=[]
    for _ in range(draws):
        idx=np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))])
        if len(np.unique(y[idx]))<2:continue
        # Direct confusion counts avoid sklearn overhead in bootstrap loops.
        predicted=p[idx]>=.5
        healthy=y[idx]==0
        unhealthy=~healthy
        values.append([(predicted==y[idx]).mean(),((~predicted[healthy]).mean()+predicted[unhealthy].mean())/2])
    a=np.asarray(values)
    return {name:np.quantile(a[:,j],[.025,.975]).tolist() for j,name in enumerate(['accuracy','balanced_accuracy'])}

def detailed_metrics(rows,p):
    y=np.asarray([int(r['label']) for r in rows])
    result=metrics(y,p)
    precision,recall,f1,support=precision_recall_fscore_support(y,p>=.5,labels=[0,1],zero_division=0)
    result['class_metrics']={name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(['Healthy','Unhealthy'])}
    result['roc_auc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    result['average_precision']=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None
    result['group_bootstrap_95_percent']=bootstrap(rows,y,p)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='evaluation')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--selection',default='selection.json')
    parser.add_argument('--skip-external',action='store_true')
    args=parser.parse_args()
    out=ROOT/args.output
    assert not (out/'metrics.json').exists(),'Completed evaluation is frozen; inspect its report instead of retesting'
    out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((ROOT/'runs'/args.selection).read_text())['selected']
    path=Path(selected['checkpoint'])
    model_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=LeafCNN(**ckpt['architecture']).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt['state_dict'])
    save_json(out/'frozen_model.json',{'checkpoint':str(path),'sha256':model_hash,'validation_selection':selected,'threshold':.5,
                                      'test_rule':'All tests evaluated after validation selection; no further tuning in this experiment'})
    rows=read_csv(args.data_dir/'manifest.csv')
    assert hashlib.sha256((args.data_dir/'manifest.csv').read_bytes()).hexdigest()==ckpt['configuration']['manifest_sha256']
    images=torch.from_numpy(np.load(args.data_dir/'images.npy')).cuda()
    results={}
    predictions=[]
    for source in sorted({r['source'] for r in rows if r['split']=='test'}):
        indices=np.array([i for i,r in enumerate(rows) if r['source']==source and r['split']=='test'])
        cohort=[rows[i] for i in indices]
        p=probabilities(model,images,indices)
        results[source]=detailed_metrics(cohort,p)
        for r,prob in zip(cohort,p):
            predictions.append({**r,'probability_unhealthy':float(prob),'predicted_class':'Unhealthy' if prob>=.5 else 'Healthy','correct':bool((prob>=.5)==int(r['label']))})
        print(source,{k:v for k,v in results[source].items() if k in ['n','accuracy','balanced_accuracy','macro_f1','confusion_matrix']},flush=True)
    del images
    torch.cuda.empty_cache()
    if not args.skip_external:
        external=read_csv(ROOT/'data/external_manifest.csv')
        indices=np.array([i for i,r in enumerate(external) if not r['exclusion']])
        cohort=[external[i] for i in indices]
        images=torch.from_numpy(np.load(ROOT/'data/external_images.npy')).cuda()
        p=probabilities(model,images,indices)
        results['TomatoVillage_external']=detailed_metrics(cohort,p)
        for r,prob in zip(cohort,p):
            predictions.append({**r,'probability_unhealthy':float(prob),'predicted_class':'Unhealthy' if prob>=.5 else 'Healthy','correct':bool((prob>=.5)==int(r['label']))})
        matched_paths={r['path'] for r in read_csv(ROOT/'data/field/manifest.csv') if r['source']=='TomatoVillage' and r['split']=='test'}
        mask=np.array([r['path'] in matched_paths for r in cohort])
        results['TomatoVillage_matched_test']=detailed_metrics([r for r,keep in zip(cohort,mask) if keep],p[mask])
        print('TomatoVillage_external',{k:v for k,v in results['TomatoVillage_external'].items() if k in ['n','accuracy','balanced_accuracy','macro_f1','confusion_matrix']},flush=True)
    write_csv(out/'predictions.csv',predictions)
    categories=[]
    for source in sorted({r['source'] for r in predictions}):
        for category in sorted({r['category'] for r in predictions if r['source']==source}):
            subset=[r for r in predictions if r['source']==source and r['category']==category]
            categories.append({'source':source,'category':category,'n':len(subset),'correct':sum(r['correct'] for r in subset),'accuracy':sum(r['correct'] for r in subset)/len(subset)})
    write_csv(out/'category_metrics.csv',categories)
    save_json(out/'metrics.json',results)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==model_hash
    print('Evaluation saved; checkpoint unchanged',flush=True)

if __name__=='__main__':
    main()
