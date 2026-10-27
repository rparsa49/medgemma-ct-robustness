"""Run paired clean/degraded CT classification with MedGemma."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from degradations import degrade


def stable_seed(seed, sample_id, kind, severity):
    digest = hashlib.sha256(f'{seed}:{sample_id}:{kind}:{severity}'.encode()).digest()
    return int.from_bytes(digest[:4], 'little')


def parse_label(text, labels):
    normalized = text.strip().casefold().strip(' .,:;!"\'')
    for label in labels:
        if normalized == label.casefold():
            return label
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    df = pd.read_csv(cfg['manifest'], dtype={'sample_id': str})
    required = {'sample_id','image_path','label'}
    if not required.issubset(df.columns): raise ValueError(f'Manifest requires {required}')
    if df.sample_id.duplicated().any(): raise ValueError('sample_id must be unique')
    if args.limit: df = df.head(args.limit)
    labels = sorted(df.label.astype(str).unique().tolist())
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    (out/'run_config.json').write_text(json.dumps({**cfg, 'labels': labels}, indent=2))
    processor = AutoProcessor.from_pretrained(cfg['model_id'])
    model = AutoModelForImageTextToText.from_pretrained(cfg['model_id'], torch_dtype='auto', device_map='auto')
    model.eval()
    prompt = cfg['prompt'].format(labels=', '.join(labels))
    def predict(image):
        messages = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k,v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=cfg['max_new_tokens'], do_sample=False)
        input_len = inputs['input_ids'].shape[-1]
        response = processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
        return response, parse_label(response, labels)
    output = out/'predictions.csv'
    existing = pd.read_csv(output, dtype={'sample_id':str}) if output.exists() else pd.DataFrame()
    completed = set(zip(existing.sample_id.astype(str), existing.degradation, existing.severity.astype(int))) if not existing.empty else set()
    for sample in tqdm(df.itertuples(index=False), total=len(df)):
        with Image.open(sample.image_path) as im: clean = im.convert('RGB')
        variants = [('clean', 0, clean)]
        for kind in cfg['degradations']:
            for severity in cfg['severity_levels']:
                variants.append((kind, severity, None))
        for kind, severity, precomputed in variants:
            if (str(sample.sample_id), kind, severity) in completed: continue
            image = precomputed if precomputed is not None else degrade(clean, kind, severity, stable_seed(seed, sample.sample_id, kind, severity))
            response, pred = predict(image)
            row = {'sample_id':str(sample.sample_id), 'image_path':sample.image_path, 'label':str(sample.label), 'degradation':kind, 'severity':severity, 'prediction':pred or '', 'raw_response':response, 'correct':int(pred == str(sample.label)), 'valid':int(pred is not None)}
            pd.DataFrame([row]).to_csv(output, mode='a', index=False, header=not output.exists())
    print(f'Predictions saved to {output}')

if __name__ == '__main__': main()
