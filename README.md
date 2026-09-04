# DR Screening — Level 1

Explainable DR severity classification: EfficientNet-B3 transfer learning,
CLAHE + illumination-normalized preprocessing, Grad-CAM explainability.

## Setup (on your laptop)

```bash
cd dr_screening
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch instead of the
default (check https://pytorch.org/get-started/locally/ for the right
command for your CUDA version) — training on CPU works but is much slower.

## Get the data

1. Download APTOS 2019 (`train.csv` + `train_images/`) from Kaggle.
2. Place them under `data/`:
   ```
   dr_screening/
     data/
       train.csv
       train_images/
         xxxxx.png
         ...
   ```
3. Open `src/config.py` and confirm `labels_csv`, `images_dir`, `image_ext`
   match your layout (defaults assume the structure above).

## Run it — button UI

```bash
streamlit run src/app.py
```

Opens in your browser. Five tabs, in order:

1. **Data Setup** — confirms your CSV/images are found, click to create the
   stratified train/val/test split (done once; test set is then locked).
2. **Train** — click to train EfficientNet-B3 from ImageNet weights. Shows
   live log output and loss/F1 curves once done.
3. **Fine-tune** — pick a checkpoint, set epochs/LR, continue training.
   Only overwrites the checkpoint if it actually improves val F1.
4. **Evaluate** — runs the trained model on the held-out test set. Shows
   accuracy, per-class precision/recall/F1, confusion matrix, referable-DR
   sensitivity/specificity/ROC-AUC, and whether the brief's targets
   (>90% sensitivity, >85% specificity) were actually met.
5. **Predict** — upload a single fundus image, get DR level, class
   probabilities, confidence, referable status, and a Grad-CAM overlay
   showing which regions drove the prediction.

## Run it — command line (equivalent, no UI)

```bash
python src/make_splits.py
python src/train.py
python src/evaluate.py --checkpoint models/dr_classifier_best.pt
python src/inference.py --image path/to/image.png --checkpoint models/dr_classifier_best.pt
python src/finetune.py --checkpoint models/dr_classifier_best.pt --epochs 10
```

## What's genuinely in scope vs. not (Level 1)

**Implemented and real:**
- Stratified split with a locked, never-peeked-at test set
- CLAHE + illumination normalization (local-average subtraction) + mild
  denoising + fundus-circle cropping
- EfficientNet-B3 transfer learning with discriminative learning rates,
  class-weighted loss, early stopping
- Grad-CAM explanations from the model's actual final conv layer
- Full evaluation suite matching the brief's requirements
- Fine-tuning workflow that only keeps improvements

**Explicitly NOT in Level 1 scope** (per the brief — don't mistake these
for missing bugs):
- Automated image reject/recapture-feedback loop (only a basic focus/
  brightness heuristic is included — flagged as non-clinical in the code)
- Structural segmentation (vessels, optic disc, microaneurysm-level
  localization) — Grad-CAM gives coarse attention, not pixel-level lesion
  segmentation
- Simulink telemedicine workflow simulation
- Multi-dataset cross-validation (IDRiD/Messidor-2) — results will only
  reflect APTOS unless you add those datasets and retrain

## On results

Nothing in this repo prints or hardcodes fabricated numbers. Every metric
in `outputs/evaluation_report.json` comes from an actual forward pass over
your test split. If sensitivity/specificity targets aren't met on your run,
`evaluate.py` will say so explicitly rather than hiding it.
