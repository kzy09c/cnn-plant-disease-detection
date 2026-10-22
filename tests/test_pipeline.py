"""Checks that protect evaluation separation and source/class balance."""
import unittest
from collections import defaultdict, Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from model import LeafCNN
from prepare_data import read_csv, ROOT

class PipelineTests(unittest.TestCase):
    def test_group_separation(self):
        manifests = [p for p in [ROOT/'data/manifest.csv',ROOT/'data/field/manifest.csv',ROOT/'data/field6/manifest.csv',ROOT/'data/field7/manifest.csv'] if p.exists()]
        if not manifests:
            self.skipTest('Local audited dataset manifests are not installed')
        for path in manifests:
            rows=read_csv(path)
            groups=defaultdict(set)
            pixels=defaultdict(set)
            for r in rows:
                if r['split']=='excluded': continue
                groups[r['merged_group']].add(r['split'])
                pixels[r['pixel_sha256']].add(r['split'])
            self.assertTrue(all(len(v)==1 for v in groups.values()))
            self.assertTrue(all(len(v)==1 for v in pixels.values()))

    def test_all_source_classes_available(self):
        if not (ROOT/'data/manifest.csv').exists():
            self.skipTest('Local audited dataset manifest is not installed')
        rows=read_csv(ROOT/'data/manifest.csv')
        for split in ['train','validation']:
            counts=Counter((r['source'],r['label']) for r in rows if r['split']==split)
            self.assertEqual(len(counts),8)
            self.assertTrue(all(v>0 for v in counts.values()))
        self.assertTrue(all(r['split']=='test' for r in rows if r['source']=='Internet16'))

    def test_custom_architectures(self):
        for attention in [False,True]:
            m=LeafCNN(attention=attention).eval()
            with torch.inference_mode():
                out=m(torch.zeros(2,3,160,160))
            self.assertEqual(out.shape,(2,2))
            self.assertTrue(torch.isfinite(out).all())
            self.assertLess(sum(p.numel() for p in m.parameters()),2_000_000)

    def test_no_test_training_draws(self):
        for file in (ROOT/'runs').glob('*/training_draws.csv'):
            counts=Counter()
            for r in read_csv(file):
                if r['split']!='train': self.assertEqual(int(r['draws']),0)
                else:counts[(r['source'],r['label'])]+=int(r['draws'])
            self.assertEqual(len(set(counts.values())),1)
            import json
            config=json.loads((file.parent/'configuration.json').read_text())
            complete=json.loads((file.parent/'complete.json').read_text())
            self.assertEqual(sum(counts.values()),config['batch_size']*config['steps_per_epoch']*complete['epochs_run'])

if __name__=='__main__': unittest.main()
