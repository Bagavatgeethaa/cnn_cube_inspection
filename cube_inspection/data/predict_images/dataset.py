import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)


# ============================================================
# Training transformations
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),

    transforms.ToTensor()
])


# ============================================================
# Validation / Test transformations
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()
])


# ============================================================
# Create datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_transform
)


# ============================================================
# Check dataset
# ============================================================

print()
print("=" * 60)
print("DATASET")
print("=" * 60)


print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)


print()
print(
    f"Classes: "
    f"{train_dataset.classes}"
)

print(
    f"Number of classes: "
    f"{len(train_dataset.classes)}"
)


# ============================================================
# Check that classes are identical
# ============================================================

if train_dataset.classes != val_dataset.classes:

    raise ValueError(
        "Training and validation classes do not match!"
    )


if train_dataset.classes != test_dataset.classes:

    raise ValueError(
        "Training and test classes do not match!"
    )


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images: {len(test_dataset)}"
)

print()

print(
    f"Classes: {train_dataset.classes}"
)

print(
    f"Class Mapping: "
    f"{train_dataset.class_to_idx}"
)

print(
    f"Number of classes: "
    f"{len(train_dataset.classes)}"
)