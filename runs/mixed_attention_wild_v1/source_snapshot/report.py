"""Build a complete local report from saved evaluations, without new inference."""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from prepare_data import ROOT,read_csv,write_csv,save_json

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def main():
    figures=ROOT/'figures'
    figures.mkdir(exist_ok=True)
    a=json.loads((ROOT/'evaluation_a/metrics.json').read_text())
    b5=json.loads((ROOT/'evaluation_b/metrics.json').read_text())
    c6=json.loads((ROOT/'evaluation_c/metrics.json').read_text())
    final_selection=json.loads((ROOT/'runs/selection_final.json').read_text())
    final_is_c=final_selection['selected']['name'].endswith('_extra_v1')
    delivered_dir='evaluation_c' if final_is_c else 'evaluation_b'
    final_label='C: six-source continuation' if final_is_c else 'B: five-source training'
    b=c6 if final_is_c else b5
    chosen=json.loads((ROOT/delivered_dir/'frozen_model.json').read_text())
    checkpoint=Path(chosen['checkpoint'])
    shutil.copyfile(checkpoint,ROOT/'best_model.pt')
    save_json(ROOT/'best_model.json',chosen)
    assert hashlib.sha256((ROOT/'best_model.pt').read_bytes()).hexdigest()==chosen['sha256']
    table=[]
    for experiment,results in [('A: four sources',a),('B: five sources',b5),('C: six-source continuation',c6)]:
        for source,m in results.items():
            table.append({'experiment':experiment,'dataset':source,**{k:v for k,v in m.items() if isinstance(v,(int,float))}})
    write_csv(ROOT/'RESULTS.csv',table)
    names=['PlantVillage','Bangladesh','TLID','PlantDoc','TomatoVillage','Internet16']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    x=np.arange(len(names))
    av=[a['TomatoVillage_matched_test' if s=='TomatoVillage' else s]['balanced_accuracy']*100 for s in names]
    bv=[b[s]['balanced_accuracy']*100 for s in names]
    ax.bar(x-.18,av,.36,label='A: four training sources',color='#8398a3')
    ax.bar(x+.18,bv,.36,label=final_label,color='#19735c')
    ax.axhline(90,linestyle='--',color='#8b4a42',linewidth=1,label='90% benchmark target')
    ax.set(xticks=x,xticklabels=['PlantVillage','Bangladesh','TLID','PlantDoc','Tomato-Village\nmatched holdout','Web photos\nweak labels, n=16'],ylabel='Balanced accuracy (%)',ylim=(0,105))
    ax.legend(loc='lower left',fontsize=9)
    ax.set_title('Custom CNNs trained from scratch: held-out benchmark results',loc='left',fontweight='bold')
    for i,(v,w) in enumerate(zip(av,bv)):
        ax.text(i-.18,v+1.2,f'{v:.1f}',ha='center',fontsize=8)
        ax.text(i+.18,w+1.2,f'{w:.1f}',ha='center',fontsize=8)
    fig.savefig(figures/'comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,s in zip(axes.flat,names):
        cm=np.asarray(b[s]['confusion_matrix'])
        ax.imshow(cm,cmap='Greens')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=15,color='white' if cm[i,j]>cm.max()/2 else '#173528')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Healthy','Unhealthy'],yticklabels=['Healthy','Unhealthy'],title=f'{s} (n={cm.sum()})',xlabel='Predicted',ylabel='Source reference')
    fig.suptitle('Delivered model: reference rows, prediction columns',fontweight='bold')
    fig.savefig(figures/'confusion_matrices.png',dpi=180)
    plt.close(fig)
    runs=[]
    for selection in ['selection.json','selection_field.json','selection_extra.json']:
        runs.extend(json.loads((ROOT/'runs'/selection).read_text())['runs'])
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for run in runs:
        history=json.loads((ROOT/'runs'/run['name']/'history.json').read_text())
        for ax,key in zip(axes,['training_loss','selection_score']):
            ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=run['name'])
    axes[0].set(title='Training loss',xlabel='Epoch',ylabel='Cross entropy with smoothing / mixup')
    axes[1].set(title='Validation selection score',xlabel='Epoch',ylabel='0.6 mean + 0.4 minimum source BA')
    axes[1].legend(fontsize=7)
    fig.savefig(figures/'training_curves.png',dpi=180)
    plt.close(fig)
    predictions=read_csv(ROOT/delivered_dir/'predictions.csv')
    web=[r for r in predictions if r['source']=='Internet16']
    fig,axes=plt.subplots(4,4,figsize=(13,12),layout='constrained')
    for ax,r in zip(axes.flat,web):
        with Image.open(r['path']) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            image.thumbnail((600,600))
            ax.imshow(image)
        color='#19735c' if r['correct']=='True' else '#a03434'
        ax.set_title(f"Reference: {r['binary_class']}\nPrediction: {r['predicted_class']} ({float(r['probability_unhealthy']):.2f} unhealthy)",fontsize=9,color=color)
        ax.axis('off')
    fig.suptitle('All 16 earlier web photos — weak reference labels, illustrative agreement only',fontsize=13,fontweight='bold')
    fig.savefig(figures/'web_predictions.png',dpi=160)
    plt.close(fig)
    lines=['# Measured results — custom tomato-leaf CNN','',
           f'All learned weights originated from random initialization in our custom CNN. Experiment C continues our own B checkpoint with joint replay. No ResNet or externally pretrained model was used. GitHub was not changed. The delivered model is {final_label}, selected on identical six-source validation cohorts before B/C test evaluation.','',
           '## Delivered model','',f"Checkpoint: `best_model.pt` ({checkpoint.stat().st_size/1024**2:.2f} MiB). Source run: `{checkpoint.parent.name}`. Decision threshold: 0.5. Labels: Healthy / Unhealthy.",'',
           '| Evaluation | Images | Accuracy | Balanced accuracy | Macro F1 | Healthy recall | Unhealthy recall |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s in names:
        m=b[s]
        lines.append(f"| {s} | {m['n']:,} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['macro_f1']:.2%} | {m['healthy_recall']:.2%} | {m['unhealthy_recall']:.2%} |")
    lines+=['','Additional checks use different evaluation units and are not included in the five-source benchmark target.','',
            '| Additional check | Images/regions | Accuracy | Balanced accuracy | Macro F1 |',
            '|---|---:|---:|---:|---:|']
    for s in ['TomatoDisease_ROI','GTLD_examples']:
        m=b[s]
        lines.append(f"| {s} | {m['n']} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['macro_f1']:.2%} |")
    lines+=['','TomatoDisease_ROI uses supplied annotation crops with context, not automatic localization in whole photographs. Its binary Healthy mapping was recovered by uniquely matching the releasing authors\' annotation count and visually checking training examples. GTLD contains just seven published examples with one Healthy photo. Neither check establishes internet-wide accuracy.']
    met=[s for s in names[:-1] if b[s]['balanced_accuracy']>=.9]
    failed=[s for s in names[:-1] if b[s]['balanced_accuracy']<.9]
    web_correct=sum(r['correct']=='True' for r in web)
    lines+=['',f"90% balanced-accuracy target met on {len(met)}/5 included benchmark sources. Below target: {', '.join(failed) if failed else 'none'}.",'',
            f"Web-photo agreement: {web_correct}/16. This is an old, small convenience sample with uncertain health labels. It does **not** estimate accuracy across random internet images. No test photos were added to training and no parameters were selected from these predictions.",'',
            '![Matched source comparison](figures/comparison.png)','',
            '## Source transfer and matched comparison','',
            '| Evaluation | A: four sources BA | B: five sources BA | C: six sources BA |',
            '|---|---:|---:|---:|']
    for s in names:
        am=a['TomatoVillage_matched_test' if s=='TomatoVillage' else s]
        lines.append(f"| {s} | {am['balanced_accuracy']:.2%} | {b5[s]['balanced_accuracy']:.2%} | {c6[s]['balanced_accuracy']:.2%} |")
    m=a['TomatoVillage_external']
    lines+=['',f"The frozen experiment-A model was also tested on all {m['n']:,} Tomato-Village images as an entirely unseen source: accuracy {m['accuracy']:.2%}, balanced accuracy {m['balanced_accuracy']:.2%}, macro F1 {m['macro_f1']:.2%}. Experiment B trained on a grouped portion of this source; its valid holdout is 1,764 images, not the entire source.",'',
            'Adding field data changes both source variety and effective sampling; this experiment does not isolate one cause of any improvement. B and C were compared on the same six-source development validation cohorts before test inference. The final model was selected using validation, not whichever model happened to score better on web photos.','',
            '![Confusion matrices](figures/confusion_matrices.png)','',
            '## Uncertainty','',
            '| Dataset | Group-bootstrap 95% interval: balanced accuracy |',
            '|---|---:|']
    for s in names:
        low,high=b[s]['group_bootstrap_95_percent']['balanced_accuracy']
        lines.append(f'| {s} | {low:.1%}–{high:.1%} |')
    lines+=['', 'Intervals resample recorded groups. They do not account for missing garden/session IDs or incorrect labels. PlantDoc has only nine Healthy test images after the overlap audit; Tomato-Village has 22. Legacy benchmark tests have been inspected in earlier work, so comparisons are exploratory. High overall accuracy can hide poor Healthy recall.','',
            '![Web-photo predictions](figures/web_predictions.png)','',
            '## Training and artifacts','',
            '![Training and validation](figures/training_curves.png)','',
            '`evaluation_a/` and `evaluation_b/` contain frozen checkpoint identities, every prediction, per-category results and full metrics. `runs/` contains configurations, validation histories and actual training draw counts. `data/` contains audited manifests, source metadata and local image caches. See `EXPERIMENT_PROTOCOL.md` for source attribution, splitting, selection and limitations.','',
            'The model expects a clear photograph of tomato foliage. It has no species detector, leaf locator, reliable out-of-distribution rejection or independently verified diagnostic ground truth. If the benchmark target or web test remains weak, the objective of dependable arbitrary internet performance remains unfinished; these results must not be called solved.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved report, figures, and best_model.pt',flush=True)

if __name__=='__main__':main()
