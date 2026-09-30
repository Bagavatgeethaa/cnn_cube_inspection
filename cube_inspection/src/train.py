import os

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from dataset import (
    train_loader,
    val_loader,
    test_loader,
)

from model import create_model

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
)


# ============================================================
# Checkpoint paths
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_checkpoint.pth"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ============================================================
# Device
# ============================================================

print()
print("=" * 60)
print("DEVICE")
print("=" * 60)

print(
    f"Using device: {DEVICE}"
)


# ============================================================
# Model
# ============================================================

model = create_model()

model = model.to(DEVICE)


# ============================================================
# Loss
# ============================================================

# CrossEntropyLoss compares the raw model logits
# with the correct labels.
#
# Loss:
# "How wrong is the prediction?"
#
# Gradient:
# "How should the weights change?"

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

# AdamW uses the gradients to update model weights.

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

# ============================================================
# Mixed precision
# ============================================================

cuda_enabled = (
    torch.cuda.is_available()
)

scaler = GradScaler(
    "cuda",
    enabled=cuda_enabled
)


# ============================================================
# Resume training
# ============================================================

start_epoch = 0

best_val_accuracy = 0.0

# ============================================================
# Training history
# ============================================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


if os.path.exists(
    LATEST_CHECKPOINT
):

    print()
    print("=" * 60)
    print("CHECKPOINT FOUND")
    print("=" * 60)

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scaler.load_state_dict(
        checkpoint[
            "scaler_state_dict"
        ]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    best_val_accuracy = checkpoint[
        "best_val_accuracy"
    ]

    print(
        f"Resuming from epoch "
        f"{start_epoch + 1}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    start_epoch,
    NUM_EPOCHS
):

    print()
    print("=" * 60)

    print(
        f"Epoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with autocast(
            device_type="cuda",
            enabled=cuda_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predictions == labels
        ).sum().item()


        progress.set_postfix(
            loss=loss.item()
        )


    # ========================================================
    # Training results
    # ========================================================

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        progress = tqdm(
            val_loader,
            desc="Validation"
        )


        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with autocast(
                device_type="cuda",
                enabled=cuda_enabled
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            val_running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predictions = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predictions == labels
            ).sum().item()


            progress.set_postfix(
                loss=loss.item()
            )


    # ========================================================
    # Validation results
    # ========================================================

    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    # ========================================================
    # Print epoch results
    # ========================================================

    print()

    print(
        f"Train Loss:     "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss:       "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy:   "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Current LR: "
        f"{scheduler.get_last_lr()[0]:.8f}"
    )


    # ========================================================
    # Save BEST model
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "best_val_accuracy":
                    best_val_accuracy,
            },
            BEST_CHECKPOINT
        )

        print()
        print(
            "New BEST model saved!"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f}"
        )


    # ========================================================
    # Save LATEST checkpoint
    # ========================================================

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,
        },
        LATEST_CHECKPOINT
    )

    print(
        "Latest checkpoint saved."
    )

    scheduler.step()

    print(
    f"Current LR: "
    f"{scheduler.get_last_lr()[0]:.8f}"
    )


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)


# ------------------------------------------------------------
# Load BEST model
# ------------------------------------------------------------

if os.path.exists(
    BEST_CHECKPOINT
):

    print(
        "Loading best model..."
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


# ------------------------------------------------------------
# Evaluation mode
# ------------------------------------------------------------

model.eval()

test_running_loss = 0.0

test_correct = 0

test_total = 0


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

with torch.no_grad():

    progress = tqdm(
        test_loader,
        desc="Testing"
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        with autocast(
            device_type="cuda",
            enabled=cuda_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        test_running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        test_total += labels.size(0)


        test_correct += (
            predictions == labels
        ).sum().item()


        progress.set_postfix(
            loss=loss.item()
        )


# ============================================================
# Test results
# ============================================================

test_loss = (
    test_running_loss / test_total
)

test_accuracy = (
    test_correct / test_total
)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test Loss:                "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy:            "
    f"{test_accuracy:.4f}"
)

print()
print(
    "Training and testing complete."
)

# # ============================================================
# # Plot training history
# # ============================================================

# import matplotlib.pyplot as plt

# epochs = range(1, len(train_losses) + 1)


# # ------------------------------------------------------------
# # Loss
# # ------------------------------------------------------------

# plt.figure(figsize=(10, 5))

# plt.plot(
#     epochs,
#     train_losses,
#     label="Training Loss"
# )

# plt.plot(
#     epochs,
#     val_losses,
#     label="Validation Loss"
# )

# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.title("Training and Validation Loss")

# plt.legend()
# plt.grid()

# plt.show()


# # ------------------------------------------------------------
# # Accuracy
# # ------------------------------------------------------------

# plt.figure(figsize=(10, 5))

# plt.plot(
#     epochs,
#     train_accuracies,
#     label="Training Accuracy"
# )

# plt.plot(
#     epochs,
#     val_accuracies,
#     label="Validation Accuracy"
# )

# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")

# plt.title("Training and Validation Accuracy")

# plt.legend()
# plt.grid()

# plt.show()

