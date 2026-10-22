"""Predict Healthy/Unhealthy for a local close-up tomato-leaf photograph."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from model import LeafCNN,CLASSES

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',type=Path,default=ROOT/'best_model.pt')
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=torch.load(args.model,map_location='cpu',weights_only=False)
    model=LeafCNN(**checkpoint['architecture']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    if device=='cuda':model.to(memory_format=torch.channels_last)
    with Image.open(args.image) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB').resize((checkpoint['image_size'],checkpoint['image_size']),Image.Resampling.BILINEAR)
        x=torch.from_numpy(np.asarray(image).copy().transpose(2,0,1)).unsqueeze(0).to(device).float()/127.5-1
    with torch.inference_mode(),torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
        p=model(x.contiguous(memory_format=torch.channels_last) if device=='cuda' else x).float().softmax(1)[0].cpu().tolist()
    print(json.dumps({'prediction':CLASSES[int(p[1]>=.5)],'probabilities':dict(zip(CLASSES,p)),
                      'note':'Scores are uncalibrated model probabilities. Assumes a close view of a tomato leaf; does not verify plant species or diagnose a disease.'},indent=2))

if __name__=='__main__':main()
