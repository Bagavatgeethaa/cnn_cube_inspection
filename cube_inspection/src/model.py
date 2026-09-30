import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)

from config import NUM_CLASSES


def create_model():

    model = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model