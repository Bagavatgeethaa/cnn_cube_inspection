from pathlib import Path
import torch

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Classes
# ============================================================

NUM_CLASSES = 2

CLASS_NAMES = [
    "bad",
    "good",
]

# ============================================================
# Training
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )