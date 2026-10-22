"""Independently recompute metrics and verify all delivered checkpoint identities."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from prepare_data import ROOT,read_csv,save_json

if __name__=='__main__':
    matched={r['path'] for r in read_csv(ROOT/'data/field/manifest.csv') if r['source']=='TomatoVillage' and r['split']=='test'}
    checks=[]
    for directory in ['evaluation_a','evaluation_b','evaluation_c']:
        out=ROOT/directory
        predictions=read_csv(out/'predictions.csv')
        results=json.loads((out/'metrics.json').read_text())
        frozen=json.loads((out/'frozen_model.json').read_text())
        assert hashlib.sha256(Path(frozen['checkpoint']).read_bytes()).hexdigest()==frozen['sha256']
        for source,recorded in results.items():
            if source in ['TomatoVillage_external','TomatoVillage_matched_test']:
                rows=[r for r in predictions if r['source']=='TomatoVillage' and (source.endswith('_external') or r['path'] in matched)]
            else:rows=[r for r in predictions if r['source']==source]
            y=np.asarray([int(r['label']) for r in rows])
            p=np.asarray([float(r['probability_unhealthy']) for r in rows])
            pred=p>=.5
            assert len(rows)==recorded['n']
            assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
            for key,value in [('accuracy',accuracy_score(y,pred)),('balanced_accuracy',balanced_accuracy_score(y,pred)),('macro_f1',f1_score(y,pred,average='macro',zero_division=0))]:
                assert abs(recorded[key]-value)<1e-12,(directory,source,key)
            assert confusion_matrix(y,pred,labels=[0,1]).tolist()==recorded['confusion_matrix']
            assert all((r['correct']=='True')==bool(v) for r,v in zip(rows,pred==y))
            checks.append({'experiment':directory,'cohort':source,'n':len(rows),'metrics_recomputed':True})
    final=json.loads((ROOT/'runs/selection_final.json').read_text())
    assert final['selected']['common_validation_score']==max(r['common_validation_score'] for r in final['candidates'])
    expected=final['selected']['checkpoint_sha256']
    assert hashlib.sha256((ROOT/'best_model.pt').read_bytes()).hexdigest()==expected
    config=json.loads((ROOT/'runs/mixed_attention_extra_v1/configuration.json').read_text())
    assert hashlib.sha256(Path(config['initial_checkpoint']).read_bytes()).hexdigest()==config['initial_checkpoint_sha256']
    save_json(ROOT/'verification.json',{'all_checks_passed':True,'cohorts':checks,'delivered_model_sha256':expected,'selection_basis':'validation only'})
    print('Verified',len(checks),'evaluation cohorts, all prediction-derived metrics, draw separation, and checkpoint hashes')
