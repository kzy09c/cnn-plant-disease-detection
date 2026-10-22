"""Evaluate the unchanged CNN on an unseen source with matched disease categories."""
from pathlib import Path
from collections import Counter
import hashlib, json, time
import numpy as np
import torch
from model import LeafCNN
from train import probabilities
from evaluate import detailed_metrics
from prepare_data import ROOT, read_csv, write_csv, save_json

DATA=ROOT/'data/3mbnb82mxd_unseen'
OUT=ROOT/'evaluation_unseen_matching'

def main():
    assert not (OUT/'metrics.json').exists(), 'Completed external evaluation is frozen'
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((DATA/'protocol_before_predictions.json').read_text())
    audit=json.loads((DATA/'audit_before_predictions.json').read_text())
    model_path=ROOT/'best_model.pt'
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_hash==protocol['checkpoint_sha256']
    assert hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest()==audit['new_manifest_sha256']
    assert hashlib.sha256((DATA/'images.npy').read_bytes()).hexdigest()==audit['image_cache_sha256']
    rows=read_csv(DATA/'manifest.csv')
    eligible=[r for r in rows if not r['exclusion']]
    assert len(eligible)==audit['eligible_images'] and len(eligible)>0
    assert Counter(r['binary_class'] for r in eligible)==audit['eligible_classes']
    assert all(r['split']=='external_test' and 'Dataset (raw)' in r['source_member'] and int(r['overlap_count'])==0 for r in eligible)
    assert {int(r['label']) for r in eligible}=={0,1}
    assert {r['category'] for r in eligible}==set(protocol['expected_category_counts'])
    identity={'checkpoint':'best_model.pt','sha256':model_hash,'threshold':.5,
              'source_url':protocol['source_url'],'manifest_sha256':audit['new_manifest_sha256'],
              'preprocessing':'EXIF-corrected RGB; bilinear resize 160x160; x/127.5-1; no special crop/background treatment.',
              'selection':'Existing delivered D checkpoint, selected before obtaining this new source. No training or adjustment.',
              'reference':'Original-image publisher categories, exact duplicates and contradictory-pixel references audited before prediction.',
           'scope_review_before_prediction':'Fixed seed source-category previews and accepted geometric duplicate QA inspected before inference. Publisher references retained. Natural-background images include some distant/partial views. This is binary Healthy/Unhealthy evaluation broken down by matched disease category, not a disease-name classifier.'}
    save_json(OUT/'frozen_model.json',identity)
    save_json(OUT/'protocol_before_predictions.json',protocol)
    save_json(OUT/'audit_before_predictions.json',audit)
    start=time.perf_counter()
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
    assert checkpoint['image_size']==160 and checkpoint['classes']==['Healthy','Unhealthy']
    model=LeafCNN(**checkpoint['architecture']).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint['state_dict'])
    images=torch.from_numpy(np.load(DATA/'images.npy')).cuda()
    indices=np.asarray([int(r['index']) for r in eligible])
    p=probabilities(model,images,indices)
    assert len(p)==len(eligible) and np.isfinite(p).all()
    result=detailed_metrics(eligible,p)
    predictions=[{**r,'probability_unhealthy':float(prob),'predicted_class':'Unhealthy' if prob>=.5 else 'Healthy',
                  'correct':bool((prob>=.5)==int(r['label']))} for r,prob in zip(eligible,p)]
    write_csv(OUT/'predictions.csv',predictions)
    categories=[]
    for category in sorted({r['category'] for r in eligible}):
        subset=[r for r in predictions if r['category']==category]
        categories.append({'category':category,'n':len(subset),'correct':sum(r['correct'] for r in subset),
                           'accuracy':sum(r['correct'] for r in subset)/len(subset)})
    write_csv(OUT/'category_metrics.csv',categories)
    result.update(source='Pakistan_matched_unseen',source_url=protocol['source_url'],
                  model_sha256=model_hash,eligible_manifest_sha256=audit['new_manifest_sha256'],
                  always_unhealthy_accuracy=sum(int(r['label']) for r in eligible)/len(eligible),
                  always_unhealthy_balanced_accuracy=.5,
                  actual_evaluation_seconds=time.perf_counter()-start)
    # Independently recompute all central metrics directly from the confusion counts.
    y=np.array([int(r['label']) for r in eligible]);pred=p>=.5
    cm=np.zeros((2,2),dtype=int)
    np.add.at(cm,(y,pred.astype(int)),1)
    accuracy=float(np.trace(cm)/cm.sum())
    recalls=cm.diagonal()/cm.sum(axis=1)
    f1s=[2*cm[k,k]/(2*cm[k,k]+cm[k,1-k]+cm[1-k,k]) for k in [0,1]]
    assert cm.tolist()==result['confusion_matrix']
    for field,expected in [('accuracy',accuracy),('balanced_accuracy',float(recalls.mean())),('macro_f1',float(np.mean(f1s)))]:
        assert abs(result[field]-expected)<1e-12
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==model_hash
    save_json(OUT/'verification.json',{'independent_confusion_count_metrics':True,'checkpoint_unchanged':True,
                                      'cohort_frozen_before_predictions':True,'photos':len(eligible),
                                      'confusion_matrix':cm.tolist(),'model_sha256':model_hash})
    save_json(OUT/'metrics.json',result)
    low,high=result['group_bootstrap_95_percent']['balanced_accuracy']
    excluded_counts={category:sum(r['category']==category and bool(r['exclusion']) for r in rows) for category in protocol['expected_category_counts']}
    lines=['# Unseen dataset with categories matching the training data','',
           f"Source: [{protocol['source_name']}, release 2]({protocol['source_url']}); Mohammad Naeem Malik, Syed Hashir Ali and Hamid Ali; CC BY 4.0.",'',
           '## What was tested','',
           'The existing delivered custom CNN was tested once without retraining. It predicts Healthy or Unhealthy. The disease categories below are publisher reference groups used to break down correct binary decisions; they are not disease names predicted by this model. All six categories are also present in the PlantVillage portion of its training sources.','',
           f"Checkpoint: `best_model.pt`; SHA-256 `{model_hash}`. Fixed threshold 0.5. Original preprocessing: EXIF-corrected RGB, bilinear resize 160x160, x/127.5-1. No test-specific crop or background adjustment.",'',
           '## Data and unseen-image audit','',
           f"The archive contains 830 raw images plus 7,200 augmented images. All raw images were considered; no augmented copies were included. After pre-prediction duplicate/overlap/reference checks, **{len(eligible)} eligible images** remained: {sum(int(r['label'])==0 for r in eligible)} Healthy and {sum(int(r['label'])==1 for r in eligible)} Unhealthy. Exclusions were decided before predictions.",'',
           f"The audit compared images with {audit['reference_records_including_parents']:,} earlier image/parent records, covering the seven training sources and earlier evaluation sources. Initial decoded-image hashes and pHash<=4 with SSIM>=.97 checks were followed by SIFT feature matching, RANSAC homography and aligned-pixel correlation checks for cropped/resized copies. Confirmed overlap families were excluded and one representative retained per remaining photo family. The final 461 images are smaller than the initial 498-image structural-audit cohort; both audits were completed before any model inference.",'',
           'In total, 357 raw entries belonged to detected photo families overlapping earlier data; another 12 were duplicate/cropped/resized copies within this release. Initial direct overlap matches included PlantDoc, Tomato-Disease parent/crop images, PlantWild and two older web examples. Therefore the whole release is not a clean independent new collection. Results below apply to the remaining images with no detected matches; undetected crops, views and session relationships can still remain.','',
           '| Reference category | Matching training category | Raw candidates | Excluded | Tested | Correct binary predictions | Incorrect | Correct rate |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in categories:
        total=protocol['expected_category_counts'][r['category']]
        lines.append(f"| {r['category']} | {protocol['training_category_mapping'][r['category']]} | {total} | {excluded_counts[r['category']]} | {r['n']} | {r['correct']} | {r['n']-r['correct']} | {r['accuracy']:.2%} |")
    lines+=['','## Overall result','',
            f"**{int(np.trace(cm))}/{len(eligible)} classified correctly; {int(cm.sum()-np.trace(cm))} incorrectly.**",'',
            '| Metric | Result |','|---|---:|',
            f"| Accuracy | {result['accuracy']:.2%} |",f"| Balanced accuracy | {result['balanced_accuracy']:.2%} |",
            f"| Macro F1 | {result['macro_f1']:.2%} |",f"| Healthy recall | {result['healthy_recall']:.2%} |",
            f"| Unhealthy recall | {result['unhealthy_recall']:.2%} |",f'| Balanced accuracy bootstrap 95% interval | {low:.2%}–{high:.2%} |','',
            'Confusion matrix: reference rows and prediction columns.','',
            '| Reference / prediction | Healthy | Unhealthy |','|---|---:|---:|',
            f'| Healthy | {cm[0,0]} | {cm[0,1]} |',f'| Unhealthy | {cm[1,0]} | {cm[1,1]} |','',
            f"Always predicting Unhealthy would score {result['always_unhealthy_accuracy']:.2%} ordinary accuracy and 50.00% balanced accuracy.",'',
            '## Interpretation and limits','',
            'This directly checks transfer to new images whose category names already occur in the training sources. Matching category names does not establish matching lighting, background, severity, acquisition or reference-label quality. The source labels are publisher references, not independent diagnoses performed here. Dataset acquisition/location claims are not independently verified. Any comparison with other datasets uses different photos and cannot isolate the causal effect of matching category names.','',
            'The per-category table reports correct Healthy/Unhealthy decisions only. A correct Unhealthy prediction on Early Blight does not mean that the model identified Early Blight by name. No additional model fitting or threshold tuning followed this test.','',
            'The repository includes individual predictions, category metrics, frozen protocol/audit records and independent verification. See dataset_sources/pakistan/README.md for source attribution, licence and download instructions. Original photographs remain in the local data/3mbnb82mxd_unseen/ cache.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
