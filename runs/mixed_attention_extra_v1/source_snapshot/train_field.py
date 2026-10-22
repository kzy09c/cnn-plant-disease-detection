"""Train the validation-selected recipe again, from scratch, on field-augmented data."""
import json
import subprocess
import sys
from prepare_data import ROOT,read_csv,save_json

if __name__=='__main__':
    selected=json.loads((ROOT/'runs/selection.json').read_text())['selected']
    recipe=selected['name'].removesuffix('_v1')
    directory=ROOT/'data/field'
    rows=read_csv(directory/'manifest.csv')
    count=len({r['source'] for r in rows if r['split']=='train'})
    batch=96 if count==6 else 100
    print('Field experiment starting from RANDOM initialization; recipe',recipe,'sources',count,flush=True)
    save_json(ROOT/'runs/field_plan.json',{'recipe':recipe,'initialization':'random','sources':count,'batch_size':batch,'max_epochs':40,'steps':64,'selection_from_experiment_a':selected})
    subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--data-dir',str(directory),'--selection','selection_field.json',
                    '--recipes',recipe,'--suffix','_field_v1','--epochs','40','--steps','64','--batch-size',str(batch)],cwd=ROOT,check=True)
