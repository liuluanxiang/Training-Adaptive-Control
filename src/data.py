"""Dataset loading, preprocessing, and reproducible train/val/test splits."""

import os

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
IMAGENET100_CLASS_FILE_SHA256 = "79f3f1552ccc478ce24bd934b4573625db584ddc1f4643ae23c994a94cb11397"
WINDOWS_WORKER_CAP_ENV = "RTC_WINDOWS_MAX_DATA_WORKERS"


def effective_num_workers(requested: int, platform: str | None = None) -> int:
    """Return a resource-safe DataLoader worker count.

    Windows gives every spawned worker a substantial commit-memory footprint and
    PyTorch transfers batches through named shared mappings.  Long queues can
    therefore hit Win32 error 1455 even while GPU memory is available.  Keep the
    requested value on other platforms, while capping Windows to two workers by
    default.  A queue retry can set ``RTC_WINDOWS_MAX_DATA_WORKERS=0`` to remove
    shared-memory multiprocessing entirely.
    """

    requested = int(requested)
    if requested < 0:
        raise ValueError(f"num_workers must be non-negative, got {requested}")
    if (platform or os.name) != "nt":
        return requested
    cap_text = os.environ.get(WINDOWS_WORKER_CAP_ENV, "2")
    try:
        cap = int(cap_text)
    except ValueError as exc:
        raise ValueError(
            f"{WINDOWS_WORKER_CAP_ENV} must be a non-negative integer, got {cap_text!r}"
        ) from exc
    if cap < 0:
        raise ValueError(f"{WINDOWS_WORKER_CAP_ENV} must be non-negative, got {cap}")
    return min(requested, cap)


def image_transform(dataset: str, train: bool, image_size: int | None = None, normalization: str = "dataset"):
    """Build training augmentations or deterministic evaluation transforms.

    Train may use random crop/flip; val/test use deterministic resize/center-crop
    so model selection is not polluted by stochastic val augmentation.
    """

    dataset = dataset.lower()
    native_sizes = {"imagenet100": 224}
    native_size = native_sizes.get(dataset, 32)
    size = int(image_size or native_size)
    norm_key = dataset if normalization == "dataset" and dataset in NORMALIZATION else normalization
    if norm_key == "dataset":
        norm_key = "imagenet"
    mean, std = NORMALIZATION[norm_key]

    if dataset == "imagenet100" or size > native_size:
        # ImageNet data and 224-input models use standard random-resized / center crop.
        spatial = [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ] if train else [
            transforms.Resize(round(size / 0.875)),
            transforms.CenterCrop(size),
        ]
    elif train:
        padding = 8 if native_size == 64 else 4
        spatial = [transforms.RandomCrop(native_size, padding=padding), transforms.RandomHorizontalFlip()]
    else:
        spatial = []
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(mean, std)])


def _split_indices(length: int, val_ratio: float, seed: int):
    """Create fixed disjoint train/val indices with a local RNG."""

    val_size = int(length * val_ratio)
    train_size = length - val_size
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[:train_size], indices[train_size:]


def _cifar_datasets(dataset: str, root: str, train_transform, eval_transform, val_ratio: float, seed: int):
    """Carve validation from the official train set; keep official test for final eval."""

    dataset_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    # Two Datasets share raw samples but use different transforms so val is not augmented.
    train_source = dataset_cls(root=root, train=True, transform=train_transform, download=True)
    validation_source = dataset_cls(root=root, train=True, transform=eval_transform, download=True)
    test_set = dataset_cls(root=root, train=False, transform=eval_transform, download=True)
    train_indices, validation_indices = _split_indices(len(train_source), val_ratio, seed)
    return (
        Subset(train_source, train_indices),
        Subset(validation_source, validation_indices),
        test_set,
        10 if dataset == "cifar10" else 100,
    )


def _resolve_imagenet100_root(root: str) -> Path:
    """Locate an ImageNet-100 root already arranged for ImageFolder."""

    supplied = Path(root)
    candidates = [supplied, supplied / "imagenet-100", supplied / "imagenet100", supplied / "ImageNet-100"]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    expected = Path(root) / "imagenet-100"
    raise FileNotFoundError(
        "ImageNet-100 was not found. Expected ImageFolder directories at "
        f"{expected / 'train'} and {expected / 'val'}."
    )


def _imagenet100_datasets(root: str, train_transform, eval_transform, val_ratio: float, seed: int):
    """Build ImageNet-100 train, internal val, and held-out official val/test."""

    dataset_root = _resolve_imagenet100_root(root)
    train_root = dataset_root / "train"
    test_root = dataset_root / "val"
    train_source = datasets.ImageFolder(train_root, transform=train_transform)
    validation_source = datasets.ImageFolder(train_root, transform=eval_transform)
    test_set = datasets.ImageFolder(test_root, transform=eval_transform)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical ImageNet-100 manifest is missing: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_classes = [str(item["wnid"]) for item in manifest.get("classes", [])]
    if manifest.get("class_file_sha256") != IMAGENET100_CLASS_FILE_SHA256:
        raise ValueError("ImageNet-100 canonical class-list hash mismatch")
    if len(train_source.classes) != 100:
        raise ValueError(
            f"ImageNet-100 train split must contain exactly 100 classes, found {len(train_source.classes)}"
        )
    if train_source.classes != test_set.classes:
        missing_from_val = sorted(set(train_source.classes) - set(test_set.classes))
        extra_in_val = sorted(set(test_set.classes) - set(train_source.classes))
        raise ValueError(
            "ImageNet-100 train/val class directories differ: "
            f"missing_from_val={missing_from_val}, extra_in_val={extra_in_val}"
        )
    if train_source.classes != expected_classes:
        raise ValueError("ImageNet-100 class order does not match the canonical manifest")
    if len(train_source) != int(manifest["train"]["images"]):
        raise ValueError("ImageNet-100 train image count does not match the canonical manifest")
    if len(test_set) != int(manifest["val"]["images"]):
        raise ValueError("ImageNet-100 val image count does not match the canonical manifest")
    train_indices, validation_indices = _split_indices(len(train_source), val_ratio, seed)
    return Subset(train_source, train_indices), Subset(validation_source, validation_indices), test_set, 100


def build_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    image_size: int | None = None,
    normalization: str = "dataset",
):
    """Build three loaders and return ``(train, val, test, num_classes)``.

    ``seed`` fixes both the subset split and the train DataLoader shuffle order.
    Val/test are not shuffled so logits/targets/metrics stay aligned.
    """

    dataset = dataset.lower()
    train_transform = image_transform(dataset, True, image_size, normalization)
    eval_transform = image_transform(dataset, False, image_size, normalization)
    if dataset in {"cifar10", "cifar100"}:
        train_set, val_set, test_set, num_classes = _cifar_datasets(
            dataset, root, train_transform, eval_transform, val_ratio, seed
        )
    elif dataset == "imagenet100":
        train_set, val_set, test_set, num_classes = _imagenet100_datasets(
            root, train_transform, eval_transform, val_ratio, seed
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    workers = effective_num_workers(num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        # CUDA training may use pinned memory for H2D copies; CPU keeps the same API.
        "pin_memory": True,
        # On Windows long queues, do not persist workers across epochs (shared-mapping growth).
        "persistent_workers": workers > 0 and os.name != "nt",
    }
    if workers > 0:
        # Default prefetch is 2; lowering it further halves Windows shared-mapping peaks.
        loader_kwargs["prefetch_factor"] = 1
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, num_classes
