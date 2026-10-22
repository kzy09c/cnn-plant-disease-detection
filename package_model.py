"""Build a portable inference bundle and check both CPU and CUDA prediction."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib
import numpy as np
from PIL import Image,ImageOps
import torch
from model import LeafCNN,CLASSES
from prepare_data import ROOT,read_csv,save_json

if __name__=='__main__':
    checkpoint=torch.load(ROOT/'best_model.pt',map_location='cpu',weights_only=False)
    cpu=LeafCNN(**checkpoint['architecture']).eval();cpu.load_state_dict(checkpoint['state_dict'])
    gpu=LeafCNN(**checkpoint['architecture']).cuda().to(memory_format=torch.channels_last).eval();gpu.load_state_dict(checkpoint['state_dict'])
    predictions=read_csv(ROOT/'evaluation_final/predictions.csv')
    sample=[]
    for label in ['0','1']:
        sample.append(next(r for r in predictions if r['source']=='PlantWild' and r['label']==label))
    smoke=[]
    for row in sample:
        with Image.open(row['path']) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB').resize((160,160),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.asarray(image).copy().transpose(2,0,1)).unsqueeze(0).float()/127.5-1
        with torch.inference_mode():p_cpu=cpu(x).softmax(1)[0].numpy()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            p_gpu=gpu(x.cuda().contiguous(memory_format=torch.channels_last)).float().softmax(1)[0].cpu().numpy()
        assert abs(float(p_cpu.sum())-1)<1e-6 and abs(float(p_gpu.sum())-1)<1e-6
        assert np.max(np.abs(p_cpu-p_gpu))<.01
        assert abs(p_gpu[1]-float(row['probability_unhealthy']))<.002
        smoke.append({'reference':row['binary_class'],'cpu_prediction':CLASSES[int(p_cpu.argmax())],'cuda_prediction':CLASSES[int(p_gpu.argmax())],
                      'max_probability_difference':float(np.max(np.abs(p_cpu-p_gpu))),'source_member':row['source_member']})
    save_json(ROOT/'inference_smoke.json',{'checks_passed':True,'examples':smoke})
    metrics=json.loads((ROOT/'evaluation_final/metrics.json').read_text())
    identity=json.loads((ROOT/'best_model.json').read_text())
    model_hash=hashlib.sha256((ROOT/'best_model.pt').read_bytes()).hexdigest()
    lines=['# Model card: custom tomato-leaf CNN','',
           f"Checkpoint: `best_model.pt`; SHA-256 `{model_hash}`.",'',
           f"Selected run: `{Path(identity['checkpoint']).parent.name}`. Parameters: {sum(p.numel() for p in cpu.parameters()):,}.",'',
           'All weights originate from our own randomly initialized sequential CNN. Later stages replay earlier training data while continuing only our own checkpoints. No ResNet, ImageNet weights, externally pretrained feature extractor or teacher model was used.','',
           'Input: close photograph of tomato foliage, EXIF orientation applied, RGB, bilinear resize to 160×160, normalization x/127.5−1. Output order: Healthy, Unhealthy; decision threshold 0.5. Scores are uncalibrated. CUDA uses float16 inference, CPU float32; small numerical differences are possible.','',
           'Unhealthy includes source-labelled disease, insects, nutrient deficiency and other visible damage. The model does not diagnose a particular disease, verify tomato species, locate leaves automatically or reliably reject unfamiliar inputs.','',
           '| Held-out check | N | Accuracy | Balanced accuracy | Macro F1 |','|---|---:|---:|---:|---:|']
    for source,m in metrics.items():lines.append(f"| {source} | {m['n']} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['macro_f1']:.2%} |")
    lines+=['','PlantWild is a newly evaluated curated internet-scene holdout, with manually screened apparently Healthy references and source-category disease references. It is not a probability sample of the internet or independent biological diagnosis. Older datasets and the 16 web photos have been inspected in earlier experiments; their comparisons are exploratory. The annotated-region test assumes supplied bounding boxes; GTLD has only seven examples. See the full local RESULTS.md and EXPERIMENT_PROTOCOL.md for source attribution, grouping, uncertainty and failures.','',
            'Training sources: PlantVillage, Bangladesh (Oni/Prama), TLID (Pellenz et al.), PlantDoc (Singh et al.), Tomato-Village (Gehlot et al.), Tomato-Disease (Chen), and PlantWild (Wei et al.). Original photos are not included in this bundle.']
    (ROOT/'MODEL_CARD.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'inference-requirements.txt').write_text('torch==2.11.0\nnumpy==2.5.2\nPillow==12.3.0\n',encoding='utf-8')
    usage='Custom CNN inference bundle\n\nTested with Python 3.12 and PyTorch 2.11.0. Install a compatible PyTorch build, NumPy and Pillow (inference-requirements.txt).\n\nRun from this folder:\n  python predict.py "C:\\path\\to\\leaf.jpg"\n\nRead MODEL_CARD.md before interpreting predictions. No datasets or original internet photos are included.\n'
    (ROOT/'INFERENCE_README.txt').write_text(usage,encoding='utf-8')
    with ZipFile(ROOT/'tomato_cnn_inference.zip','w',ZIP_DEFLATED) as bundle:
        for name in ['best_model.pt','best_model.json','model.py','predict.py','inference-requirements.txt','MODEL_CARD.md','INFERENCE_README.txt']:
            bundle.write(ROOT/name,name)
    print('Verified CPU/CUDA inference and created portable bundle',flush=True)
