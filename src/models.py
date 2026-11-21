"""Model factory for CNN, ConvNeXt, and DeiT classifiers."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    name: str,
    num_classes: int,
    pretrained_checkpoint: str | None = None,
    image_size: int | None = None,
):
    """Build a classifier and return an ``nn.Module`` not yet moved to device.

    Args:
        name: Model id from the config.
        num_classes: Dataset class count for the final head.
        pretrained_checkpoint: Local DeiT checkpoint path; CNN/ConvNeXt train from scratch.

    ResNet/VGG stems are adapted to keep spatial detail on 32x32 or 64x64 inputs;
    DeiT keeps the official 224x224 structure for strict pretrained loading.
    """

    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        # Default ImageNet 7x7/stride-2 stem is too aggressive for small images; use CIFAR stem.
        if image_size is None or image_size < 128:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model

    if name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
        # Same small-image input protocol as ResNet-18 for fair comparisons.
        if image_size is None or image_size < 128:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model

    if name == "vgg16":
        model = models.vgg16_bn(weights=None, num_classes=num_classes)
        # Adaptive pooling maps any input size to 1x1; head input is fixed at 512-d.
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        return model

    if name == "convnext_tiny":
        # Keep official torchvision ConvNeXt-Tiny; replace only the dataset head.
        return models.convnext_tiny(weights=None, num_classes=num_classes)

    if name == "deit_tiny_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("DeiT requires timm; install requirements.txt in the project environment.") from exc
        # Build target-class model first, then load local weights for auditable provenance.
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        if pretrained_checkpoint:
            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"DeiT checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = dict(checkpoint.get("model", checkpoint))
            # Official head is ImageNet-1K; drop it and keep the new dataset head.
            state.pop("head.weight", None)
            state.pop("head.bias", None)
            incompatible = model.load_state_dict(state, strict=False)
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            # Any missing/unexpected key outside the head means backbone/structure mismatch.
            if unexpected or not missing.issubset({"head.weight", "head.bias"}):
                raise RuntimeError(
                    f"Incompatible DeiT checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        return model

    raise ValueError(f"Unsupported model: {name}")
