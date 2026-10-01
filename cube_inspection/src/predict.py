import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# LOCAL IMPORTS
# ============================================================

# Ensure local imports (model.py, config.py) resolve properly
# when compiled or executed from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

GOOD_FOLDER = GROUND_TRUTH_DIR / "GOOD"

BAD_FOLDER = GROUND_TRUTH_DIR / "BAD"

CONFIDENCE_THRESHOLD = 0.70

IMAGE_SIZE = 384

SUPPORTED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]

MISCLASSIFIED_DIR = Path("misclassified")

FN_DIR = (
    MISCLASSIFIED_DIR /
    "actual_bad_predicted_good"
)

FP_DIR = (
    MISCLASSIFIED_DIR /
    "actual_good_predicted_bad"
)

RESULTS_CSV = Path("prediction_results.csv")

CONFUSION_MATRIX_IMAGE = Path(
    "confusion_matrix.png"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CUBE INSPECTION")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
)


# ============================================================
# VALIDATE CLASS NAMES
# ============================================================

if len(CLASS_NAMES) != 2:

    raise ValueError(
        "This script expects exactly 2 classes "
        f"(BAD and GOOD), but CLASS_NAMES contains: "
        f"{CLASS_NAMES}"
    )

# Find BAD and GOOD class indexes from config.
CLASS_NAME_MAP = {
    str(name).strip().upper(): index
    for index, name in enumerate(CLASS_NAMES)
}

if "BAD" not in CLASS_NAME_MAP:
    raise ValueError(
        f"'BAD' class not found in CLASS_NAMES: "
        f"{CLASS_NAMES}"
    )

if "GOOD" not in CLASS_NAME_MAP:
    raise ValueError(
        f"'GOOD' class not found in CLASS_NAMES: "
        f"{CLASS_NAMES}"
    )

BAD_CLASS_INDEX = CLASS_NAME_MAP["BAD"]

GOOD_CLASS_INDEX = CLASS_NAME_MAP["GOOD"]


# ============================================================
# TRANSFORM
# ============================================================

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

if not MODEL_PATH.exists():

    print(
        f"\nERROR: Model checkpoint not found:"
        f"\n{MODEL_PATH}"
    )

    raise SystemExit(1)


print(
    f"Loading trained model:\n{MODEL_PATH}"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Support checkpoints containing model_state_dict.
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Some checkpoints are simply state_dict dictionaries.
        state_dict = checkpoint

else:

    raise ValueError(
        "Unsupported checkpoint format."
    )


model.load_state_dict(
    state_dict
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GROUND TRUTH
# ============================================================

ground_truth = {}


# ------------------------------------------------------------
# GOOD = 1
# ------------------------------------------------------------

if GOOD_FOLDER.exists():

    for file in GOOD_FOLDER.iterdir():

        if file.is_file():

            ground_truth[file.name] = 1


# ------------------------------------------------------------
# BAD = 0
# ------------------------------------------------------------

if BAD_FOLDER.exists():

    for file in BAD_FOLDER.iterdir():

        if file.is_file():

            ground_truth[file.name] = 0


print(
    f"\nLoaded "
    f"{len(ground_truth)} "
    f"ground-truth labels"
)

print(
    f"GOOD folder: {GOOD_FOLDER}"
)

print(
    f"BAD folder : {BAD_FOLDER}"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STORAGE
# ============================================================

results = []

y_true = []

y_pred = []

uncertain_results = []


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict(image_path):
    """
    Predict BAD / GOOD for one image.

    Returns:
        class_name
        confidence
        bad_probability
        good_probability
        predicted_class_index
    """

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

    confidence_tensor, predicted_class = (
        torch.max(
            probabilities,
            dim=1
        )
    )

    predicted_class_index = (
        predicted_class.item()
    )

    confidence = (
        confidence_tensor.item()
    )

    class_name = (
        str(
            CLASS_NAMES[
                predicted_class_index
            ]
        )
        .upper()
    )

    bad_probability = (
        probabilities[
            0,
            BAD_CLASS_INDEX
        ].item()
    )

    good_probability = (
        probabilities[
            0,
            GOOD_CLASS_INDEX
        ].item()
    )

    return (
        class_name,
        confidence,
        bad_probability,
        good_probability,
        predicted_class_index,
    )


# ============================================================
# FIND IMAGES
# ============================================================

folder = Path(IMAGE_FOLDER)

if not folder.exists():

    print(
        f"\nERROR: Prediction folder not found:"
        f"\n{folder}"
    )

    raise SystemExit(1)


images = [
    file
    for file in folder.iterdir()
    if (
        file.is_file()
        and file.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )
]


if len(images) == 0:

    print(
        "\nNo images found in:"
        f"\n{folder}"
    )

    raise SystemExit(0)


images = sorted(images)


print(
    f"\nFound {len(images)} images"
)

print()


# ============================================================
# RUN PREDICTIONS
# ============================================================

for index, image_path in enumerate(images):

    try:

        (
            class_name,
            confidence,
            bad_probability,
            good_probability,
            predicted_class_index,
        ) = predict(image_path)


        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        is_uncertain = (
            confidence <
            CONFIDENCE_THRESHOLD
        )


        if is_uncertain:

            prediction_status = "UNCERTAIN"

        else:

            prediction_status = (
                class_name
            )


        # ----------------------------------------------------
        # PRINT PREDICTION
        # ----------------------------------------------------

        print("-" * 60)

        print(
            f"Image      : "
            f"{image_path.name}"
        )

        print(
            f"BAD        : "
            f"{bad_probability * 100:.2f}%"
        )

        print(
            f"GOOD       : "
            f"{good_probability * 100:.2f}%"
        )

        print(
            f"PREDICTION : "
            f"{prediction_status}"
        )

        print(
            f"CONFIDENCE : "
            f"{confidence * 100:.2f}%"
        )


        if is_uncertain:

            print(
                "STATUS     : "
                "Below confidence threshold"
            )


        # ====================================================
        # GROUND TRUTH EVALUATION
        # ====================================================

        if image_path.name not in ground_truth:

            print(
                "GROUND TRUTH: "
                "Not available"
            )

            continue


        actual_label = (
            ground_truth[
                image_path.name
            ]
        )


        actual_class_name = (
            "BAD"
            if actual_label == 0
            else "GOOD"
        )


        print(
            f"ACTUAL     : "
            f"{actual_class_name}"
        )


        # ----------------------------------------------------
        # UNCERTAIN PREDICTIONS
        # ----------------------------------------------------

        if is_uncertain:

            uncertain_results.append({
                "filename":
                    image_path.name,

                "actual":
                    actual_class_name,

                "predicted":
                    "UNCERTAIN",

                "bad_probability":
                    bad_probability,

                "good_probability":
                    good_probability,

                "confidence":
                    confidence,
            })

            continue


        # ====================================================
        # BINARY PREDICTION
        # ====================================================

        predicted_label = (
            0
            if predicted_class_index
            == BAD_CLASS_INDEX
            else 1
        )


        correct = (
            actual_label ==
            predicted_label
        )


        # ----------------------------------------------------
        # METRICS STORAGE
        # ----------------------------------------------------

        y_true.append(
            actual_label
        )

        y_pred.append(
            predicted_label
        )


        results.append({

            "filename":
                image_path.name,

            "actual":
                actual_class_name,

            "predicted":
                class_name,

            "bad_probability":
                bad_probability,

            "good_probability":
                good_probability,

            "confidence":
                confidence,

            "correct":
                correct,

        })


        # ----------------------------------------------------
        # MISCLASSIFIED IMAGES
        # ----------------------------------------------------

        if not correct:

            # -----------------------------------------------
            # FALSE NEGATIVE
            #
            # Actual BAD -> Predicted GOOD
            # -----------------------------------------------

            if (
                actual_label == 0
                and predicted_label == 1
            ):

                destination = (
                    FN_DIR /
                    (
                        f"{index:05d}_"
                        f"{image_path.name}"
                    )
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                print(
                    "MISCLASSIFIED: "
                    "BAD -> GOOD"
                )


            # -----------------------------------------------
            # FALSE POSITIVE
            #
            # Actual GOOD -> Predicted BAD
            # -----------------------------------------------

            elif (
                actual_label == 1
                and predicted_label == 0
            ):

                destination = (
                    FP_DIR /
                    (
                        f"{index:05d}_"
                        f"{image_path.name}"
                    )
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                print(
                    "MISCLASSIFIED: "
                    "GOOD -> BAD"
                )


    except Exception as error:

        print(
            f"{image_path.name} -> "
            f"ERROR: {error}"
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("EVALUATION")
print("=" * 60)


# ============================================================
# SAVE UNCERTAIN RESULTS
# ============================================================

if len(uncertain_results) > 0:

    uncertain_df = pd.DataFrame(
        uncertain_results
    )

    uncertain_df.to_csv(
        "uncertain_predictions.csv",
        index=False
    )

    print(
        f"\nUncertain predictions: "
        f"{len(uncertain_results)}"
    )

    print(
        "Saved uncertain_predictions.csv"
    )


# ============================================================
# CONFUSION MATRIX / METRICS
# ============================================================

if len(y_true) == 0:

    print(
        "\nNo labeled images were available "
        "for evaluation."
    )

else:

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        RESULTS_CSV,
        index=False
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    #
    # labels:
    # 0 = BAD
    # 1 = GOOD
    #
    #               Predicted
    #               BAD   GOOD
    # Actual BAD    TP    FN
    #        GOOD   FP    TN
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


    TP = cm[0, 0]

    FN = cm[0, 1]

    FP = cm[1, 0]

    TN = cm[1, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )


    # ========================================================
    # PRINT CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        "             Predicted"
    )

    print(
        "             BAD    GOOD"
    )

    print(
        f"Actual BAD  "
        f"{TP:5d}  "
        f"{FN:5d}"
    )

    print(
        f"Actual GOOD "
        f"{FP:5d}  "
        f"{TN:5d}"
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print()

    print(
        f"TP (BAD->BAD)  : {TP}"
    )

    print(
        f"FN (BAD->GOOD) : {FN}"
    )

    print(
        f"FP (GOOD->BAD) : {FP}"
    )

    print(
        f"TN (GOOD->GOOD): {TN}"
    )

    print()

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "BAD",
                "GOOD"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "BAD",
            "GOOD"
        ],
        yticklabels=[
            "BAD",
            "GOOD"
        ]
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "CNN Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        CONFUSION_MATRIX_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame([{

        "total_images":
            len(images),

        "labeled_images":
            len(y_true),

        "uncertain_images":
            len(uncertain_results),

        "evaluated_images":
            len(y_true),

        "true_positive_bad":
            TP,

        "false_negative_bad":
            FN,

        "false_positive_bad":
            FP,

        "true_negative_good":
            TN,

        "accuracy":
            accuracy,

        "precision_bad":
            precision,

        "recall_bad":
            recall,

        "f1_bad":
            f1,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

    }])


    summary.to_csv(
        "evaluation_summary.csv",
        index=False
    )


    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print()
    print(
        "Saved prediction_results.csv"
    )

    print(
        "Saved evaluation_summary.csv"
    )

    print(
        "Saved confusion_matrix.png"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

misclassified_count = sum(
    1
    for item in results
    if not item["correct"]
)

print()
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print(
    f"Total images       : {len(images)}"
)

print(
    f"Evaluated images   : {len(y_true)}"
)

print(
    f"Uncertain images   : {len(uncertain_results)}"
)

print(
    f"Misclassified      : {misclassified_count}"
)

print(
    f"Model              : {MODEL_PATH}"
)

print(
    f"Prediction folder  : {IMAGE_FOLDER}"
)

print("=" * 60)
