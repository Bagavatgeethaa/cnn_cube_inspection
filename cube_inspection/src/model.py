import torch.nn as nn
from torchvision.models import efficientnet_v2_s

from config import NUM_CLASSES


def create_model():

    model = efficientnet_v2_s(
        weights=None
    )

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model