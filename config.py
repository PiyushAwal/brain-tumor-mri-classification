"""Central configuration for the Brain Tumor MRI Classification project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"              # data/train, data/valid (optional), data/test
MODELS_DIR = ROOT / "saved_models"    # trained .h5 files land here
REPORTS_DIR = ROOT / "reports"        # metrics, histories, comparison.csv
CKPT_DIR = MODELS_DIR / "checkpoints"  # best-epoch checkpoints (.keras), exported to .h5 after training

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15                      # used only when the dataset has no validation folder

EPOCHS = 25                           # phase 1 (custom CNN: full training / transfer: head only)
FINE_TUNE_EPOCHS = 10                 # phase 2 (transfer models: top layers unfrozen)
UNFREEZE_LAYERS = 30
PATIENCE = 6

MODEL_NAMES = ["custom_cnn", "resnet50", "mobilenetv2", "inceptionv3", "efficientnetb0"]

TRAIN_DIR_NAMES = {"train", "training"}
VAL_DIR_NAMES = {"valid", "val", "validation"}
TEST_DIR_NAMES = {"test", "testing"}

for _d in (MODELS_DIR, REPORTS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
