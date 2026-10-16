# NeuroScan - Brain Tumor MRI Image Classification

Custom CNN + transfer learning (ResNet50, MobileNetV2, InceptionV3, EfficientNetB0) with an
interactive Streamlit dashboard: dataset explorer, training curves, model comparison,
and live diagnosis with Grad-CAM explanations.

## Setup (Python 3.10 - 3.13, TensorFlow 2.20+ / Keras 3)
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Download the dataset (link in the project brief) and arrange it as:
```
data/
  train/<class>/*.jpg
  valid/<class>/*.jpg   # optional; 15% of train is used if missing
  test/<class>/*.jpg
```
Check it with `python data_utils.py`.

## Train, evaluate, run
```bash
python train.py                        # all 5 models -> saved_models/*.h5, reports/*.json
python train.py --models custom_cnn    # a single model
python evaluate.py                     # re-evaluate saved models
streamlit run app.py                   # dashboard
```
Tip: run `python train.py --epochs 2 --fine-tune-epochs 1 --models custom_cnn` first as a smoke test.

## Structure
| File | Purpose |
|---|---|
| config.py | Paths and hyperparameters |
| data_utils.py | Loading, 0-1 normalization, augmentation, class weights |
| model_zoo.py | Custom CNN and transfer models |
| train.py | Two-phase training with EarlyStopping / ModelCheckpoint |
| evaluate.py | Accuracy, precision, recall, F1, confusion matrix, latency |
| gradcam.py | Explainability heatmaps |
| app.py | Streamlit dashboard |

## Disclaimer
Research prototype. Not a certified medical device.
