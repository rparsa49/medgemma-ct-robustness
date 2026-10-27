# MedGemma Robustness Under CT Image Degradation

Reproducible evaluation of MedGemma classification consistency under seven display-domain CT perturbations at five severities, relative to a clean baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

Request/accept the model's terms on Hugging Face before downloading. Use an appropriately provisioned GPU; the model may not fit in CPU memory or smaller GPUs.

## Dataset

Supply your own appropriately licensed, **de-identified** CT slices as RGB/PNG images, with consistent CT windowing. Create `data/manifest.csv` with `sample_id,image_path,label` (see `data/manifest.example.csv`). Labels must correspond to image-level, visually supportable findings; do not use study-level labels indiscriminately on arbitrary slices. This repository does not bundle clinical data. Use a fixed held-out test cohort and keep patient identities disjoint from development data.

## Run

```bash
python src/run.py --config configs/default.yaml --limit 5
python src/run.py --config configs/default.yaml
python src/evaluate.py --predictions results/predictions.csv
```

The pipeline saves per-image model responses, parsed predictions, group metrics, and a robustness figure. It resumes completed conditions in `predictions.csv`; delete that file before changing the manifest, model, prompt, or config. All seven perturbations operate on already-windowed display images, **not** raw CT Hounsfield units; JPEG and bias field are stress tests rather than validated CT acquisition physics simulations.

## Experimental design

- 1 clean + 7 × 5 degraded images per CT slice (36 inferences/image).
- Fixed zero-shot label selection prompt and greedy decoding.
- Accuracy and macro-F1 count unparsable responses as incorrect; valid-rate tracks those separately.
- Paired prediction consistency compares each degraded image with its own clean prediction, whether or not that baseline prediction is correct.
- Severity values are heuristic and not calibrated to scanner physics or clinical acceptability.

## Limitations

A 2D slice cannot represent a complete CT study. Report dataset provenance, windowing, label reliability, per-class results, patient-level uncertainty intervals, and failure examples before drawing clinical conclusions. This is a research benchmark, not a diagnostic device.
