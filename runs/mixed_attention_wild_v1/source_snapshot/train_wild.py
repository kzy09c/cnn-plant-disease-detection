"""One final continuation adding curated internet scenes, with all-source replay."""
import json,subprocess,sys
from prepare_data import ROOT,save_json

if __name__=='__main__':
    initial=json.loads((ROOT/'runs/selection_final.json').read_text())['selected']
    save_json(ROOT/'runs/wild_plan.json',{'initial_checkpoint':initial['checkpoint'],'initialization':'Only our own previously trained custom CNN',
        'dataset':'Seven sources; filtered PlantWild added after prior exploratory results','epochs':25,'steps':64,'batch_size':112,'lr_multiplier':.3,
        'selection':'Compare C and D on the same seven development-validation cohorts; fixed .5 decisions; no PlantWild test prediction before selection'})
    subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--data-dir',str(ROOT/'data/field7'),
                    '--selection','selection_wild_run.json','--recipes','mixed_attention','--suffix','_wild_v1',
                    '--initial-checkpoint',initial['checkpoint'],'--lr-multiplier','.3','--epochs','25','--steps','64','--batch-size','112'],cwd=ROOT,check=True)
