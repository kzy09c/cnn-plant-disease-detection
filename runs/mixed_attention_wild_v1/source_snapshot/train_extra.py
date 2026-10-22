"""Jointly add audited regions while replaying all earlier training sources."""
import json,subprocess,sys
from prepare_data import ROOT,save_json

if __name__=='__main__':
    selected=json.loads((ROOT/'runs/selection_field.json').read_text())['selected']
    recipe=selected['name'].removesuffix('_field_v1')
    save_json(ROOT/'runs/extra_plan.json',{'initial_checkpoint':selected['checkpoint'],
        'plan':'Continue only this project own CNN on all six training sources with balanced replay; no external pretrained weights',
        'recipe':recipe,'epochs':25,'steps_per_epoch':64,'batch_size':96,'lr_multiplier':.3,
        'selection':'Compare B and C on the same six validation cohorts before their test evaluation'})
    subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--data-dir',str(ROOT/'data/field6'),
                    '--selection','selection_extra.json','--recipes',recipe,'--suffix','_extra_v1',
                    '--initial-checkpoint',selected['checkpoint'],'--lr-multiplier','.3',
                    '--epochs','25','--steps','64','--batch-size','96'],cwd=ROOT,check=True)
