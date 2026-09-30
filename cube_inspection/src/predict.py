import os
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import create_model

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    CLASS_NAMES,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

IMAGE_FOLDER = DATA_DIR / "predict_images"

CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CUBE INSPECTION")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# TRANSFORM
# ============================================================
IMAGE_SIZE = 384

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# MODEL
# ============================================================

print("\nCreating model...")

model = create_model()

model = model.to(DEVICE)

# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print(f"\nERROR: {MODEL_PATH} not found")
    raise SystemExit

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    class_id = predicted_class.item()

    confidence = confidence.item()

    class_name = CLASS_NAMES[class_id]

    bad_probability = probabilities[0][0].item()

    good_probability = probabilities[0][1].item()

    return (
        class_name,
        confidence,
        bad_probability,
        good_probability
    )

# ============================================================
# FIND IMAGES
# ============================================================

folder = Path(IMAGE_FOLDER)

if not folder.exists():

    print(
        f"\nERROR: Folder not found: {folder}"
    )

    raise SystemExit

images = [
    file
    for file in folder.iterdir()
    if file.suffix.lower()
    in [".jpg", ".jpeg", ".png",
        ".bmp", ".webp"]
]

if len(images) == 0:

    print("\nNo images found.")
    raise SystemExit

print(f"\nFound {len(images)} images\n")

# ============================================================
# RUN PREDICTIONS
# ============================================================

for image_path in sorted(images):

    try:

        (
            class_name,
            confidence,
            bad_probability,
            good_probability
        ) = predict(image_path)

        print("-" * 60)

        print(
            f"Image : {image_path.name}"
        )

        print(
            f"BAD  : {bad_probability * 100:.2f}%"
        )

        print(
            f"GOOD : {good_probability * 100:.2f}%"
        )

        print(
            f"PREDICTION : {class_name}"
        )

        print(
            f"CONFIDENCE : {confidence * 100:.2f}%"
        )

    except Exception as error:

        print(
            f"{image_path.name} -> ERROR: {error}"
        )

print()
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)