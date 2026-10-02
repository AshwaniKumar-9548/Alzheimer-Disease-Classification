import copy
import hashlib
import os
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import sympy
import sympy.printing
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainingConfig:
    train_dir: str = "Dataset/Training Set"
    test_dir: str = "Dataset/Testing Set"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 18
    val_size: float = 0.15
    backbone_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 5e-4
    label_smoothing: float = 0.08
    dropout: float = 0.5
    random_erasing_prob: float = 0.15
    use_weighted_sampler: bool = True
    sampler_power: float = 0.5
    class_weight_power: float = 0.5
    freeze_backbone_epochs: int = 2
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    split_by_hash_groups: bool = True
    deduplicate_train: bool = True
    audit_dataset: bool = True
    seed: int = 42
    num_workers: int = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)


class TransformedImageFolderSubset(Dataset):
    def __init__(self, samples, classes, class_to_idx, transform=None):
        self.samples = list(samples)
        self.targets = [label for _, label in self.samples]
        self.classes = list(classes)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(img_size, random_erasing_prob=0.15):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def _class_counts(labels, num_classes):
    return np.bincount(labels, minlength=num_classes)


def _compute_file_hash(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_inferred_patient_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r" \(\d+\)$", "", stem)


def _extract_inferred_patient_group_key(path):
    class_name = os.path.basename(os.path.dirname(path))
    patient_id = _extract_inferred_patient_id(path)
    return f"{class_name}::{patient_id}"


def _build_hash_records(samples, description):
    records = []
    for path, label in tqdm(samples, desc=description, leave=False):
        records.append(
            {
                "path": path,
                "label": label,
                "hash": _compute_file_hash(path),
            }
        )
    return records


def _summarize_hash_records(train_records, test_records=None):
    train_by_hash = {}
    for record in train_records:
        train_by_hash.setdefault(record["hash"], []).append(record)

    duplicate_groups = [group for group in train_by_hash.values() if len(group) > 1]
    duplicate_files = sum(len(group) for group in duplicate_groups)
    redundant_duplicate_files = sum(len(group) - 1 for group in duplicate_groups)
    cross_class_duplicate_groups = sum(
        1 for group in duplicate_groups if len({item["label"] for item in group}) > 1
    )

    summary = {
        "train_duplicate_groups": len(duplicate_groups),
        "train_duplicate_files": duplicate_files,
        "train_redundant_duplicate_files": redundant_duplicate_files,
        "train_cross_class_duplicate_groups": cross_class_duplicate_groups,
        "train_test_exact_hash_overlap": 0,
    }

    if test_records is not None:
        test_hashes = {record["hash"] for record in test_records}
        summary["train_test_exact_hash_overlap"] = sum(
            1 for hash_value in train_by_hash if hash_value in test_hashes
        )

    return summary


def _summarize_inferred_patient_groups(samples, class_names):
    summary = []

    for class_index, class_name in enumerate(class_names):
        class_paths = [path for path, label in samples if label == class_index]
        patient_counts = {}
        for path in class_paths:
            patient_id = _extract_inferred_patient_id(path)
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1

        repeated_groups = sum(1 for count in patient_counts.values() if count > 1)
        max_group_size = max(patient_counts.values(), default=0)
        mean_group_size = (
            float(len(class_paths)) / float(len(patient_counts))
            if patient_counts
            else 0.0
        )

        summary.append(
            {
                "class_name": class_name,
                "num_images": len(class_paths),
                "num_patients": len(patient_counts),
                "repeated_patient_groups": repeated_groups,
                "mean_slices_per_patient": mean_group_size,
                "max_slices_per_patient": max_group_size,
            }
        )

    return summary


def _split_records_by_hash(records, val_size, seed, deduplicate_train):
    random_state = random.Random(seed)
    records_by_label = {}
    for record in records:
        records_by_label.setdefault(record["label"], []).append(record)

    train_samples = []
    val_samples = []

    for label_records in records_by_label.values():
        groups_by_hash = {}
        for record in label_records:
            groups_by_hash.setdefault(record["hash"], []).append(record)

        hash_groups = list(groups_by_hash.values())
        random_state.shuffle(hash_groups)

        if deduplicate_train:
            target_val_groups = max(1, int(round(len(hash_groups) * val_size)))
            val_groups = hash_groups[:target_val_groups]
            train_groups = hash_groups[target_val_groups:]

            if not train_groups and val_groups:
                train_groups = val_groups[-1:]
                val_groups = val_groups[:-1]
        else:
            target_val_files = max(1, int(round(len(label_records) * val_size)))
            val_groups = []
            train_groups = []
            current_val_files = 0

            for group in hash_groups:
                if current_val_files < target_val_files:
                    val_groups.append(group)
                    current_val_files += len(group)
                else:
                    train_groups.append(group)

            if not train_groups and val_groups:
                train_groups = val_groups[-1:]
                val_groups = val_groups[:-1]

        for group in train_groups:
            if deduplicate_train:
                item = group[0]
                train_samples.append((item["path"], item["label"]))
            else:
                train_samples.extend((item["path"], item["label"]) for item in group)

        for group in val_groups:
            if deduplicate_train:
                item = group[0]
                val_samples.append((item["path"], item["label"]))
            else:
                val_samples.extend((item["path"], item["label"]) for item in group)

    return train_samples, val_samples


def prepare_dataloaders(config):
    train_transform, eval_transform = build_transforms(
        config.img_size,
        random_erasing_prob=config.random_erasing_prob,
    )

    full_train_ds = datasets.ImageFolder(config.train_dir)
    test_base_ds = datasets.ImageFolder(config.test_dir)

    if full_train_ds.classes != test_base_ds.classes:
        raise ValueError("Train and test folders do not share the same class order.")

    class_names = full_train_ds.classes
    audit_summary = None

    if config.split_by_hash_groups or config.deduplicate_train or config.audit_dataset:
        train_records = _build_hash_records(full_train_ds.samples, "Hashing train images")
    else:
        train_records = None

    if config.audit_dataset:
        test_records = _build_hash_records(test_base_ds.samples, "Hashing test images")
    else:
        test_records = None

    if config.split_by_hash_groups or config.deduplicate_train:
        train_samples, val_samples = _split_records_by_hash(
            train_records,
            val_size=config.val_size,
            seed=config.seed,
            deduplicate_train=config.deduplicate_train,
        )
    else:
        labels = np.array([label for _, label in full_train_ds.samples])
        indices = np.arange(len(full_train_ds.samples))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=config.val_size,
            stratify=labels,
            random_state=config.seed,
        )

        train_samples = [full_train_ds.samples[i] for i in train_indices]
        val_samples = [full_train_ds.samples[i] for i in val_indices]

    train_ds = TransformedImageFolderSubset(
        samples=train_samples,
        classes=class_names,
        class_to_idx=full_train_ds.class_to_idx,
        transform=train_transform,
    )
    val_ds = TransformedImageFolderSubset(
        samples=val_samples,
        classes=class_names,
        class_to_idx=full_train_ds.class_to_idx,
        transform=eval_transform,
    )
    test_ds = TransformedImageFolderSubset(
        samples=test_base_ds.samples,
        classes=class_names,
        class_to_idx=test_base_ds.class_to_idx,
        transform=eval_transform,
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_counts = _class_counts(train_ds.targets, len(class_names))
    val_counts = _class_counts(val_ds.targets, len(class_names))
    test_counts = _class_counts(test_ds.targets, len(class_names))

    train_sampler = None
    if config.use_weighted_sampler:
        train_targets = np.asarray(train_ds.targets)
        sample_weights = np.power(train_counts[train_targets].astype(np.float64), -config.sampler_power)
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    class_weights = np.power(train_counts.astype(np.float32), -config.class_weight_power)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if train_records is not None:
        audit_summary = _summarize_hash_records(train_records, test_records=test_records)
        hash_by_path = {record["path"]: record["hash"] for record in train_records}
        train_hashes = {hash_by_path[path] for path, _ in train_samples}
        val_hashes = {hash_by_path[path] for path, _ in val_samples}
        audit_summary["train_val_exact_hash_overlap"] = len(train_hashes & val_hashes)
        audit_summary["train_samples_after_split"] = len(train_samples)
        audit_summary["val_samples_after_split"] = len(val_samples)
        audit_summary["test_patient_summary"] = _summarize_inferred_patient_groups(
            test_base_ds.samples,
            class_names,
        )

    return {
        "class_names": class_names,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "class_weights": class_weights,
        "audit_summary": audit_summary,
    }


def build_model(num_classes, dropout):
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model, trainable):
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def create_scaler(device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    try:
        return torch.autocast(device_type=device.type, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        metrics["macro_auc_ovr"] = roc_auc_score(
            y_true_bin,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["macro_auc_ovr"] = float("nan")

    return metrics


def _optimizer_parameters(optimizer):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                yield parameter


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    description="Epoch",
    gradient_clip_norm=None,
):
    is_training = optimizer is not None
    amp_enabled = scaler is not None and device.type == "cuda"

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    progress = tqdm(loader, desc=description, leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with autocast_context(device, amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(_optimizer_parameters(optimizer)),
                            gradient_clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            list(_optimizer_parameters(optimizer)),
                            gradient_clip_norm,
                        )
                    optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, y_prob.shape[1])
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, y_true, y_pred, y_prob


def train_model(model, loaders, config, device):
    class_weights = loaders["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )

    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": config.backbone_lr},
            {"params": model.classifier.parameters(), "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.reduce_lr_patience,
    )
    scaler = create_scaler(device)

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    backbone_frozen = config.freeze_backbone_epochs > 0
    if backbone_frozen:
        set_backbone_trainable(model, trainable=False)
        print(f"Backbone frozen for the first {config.freeze_backbone_epochs} epoch(s).")

    for epoch in range(1, config.epochs + 1):
        if backbone_frozen and epoch == config.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            backbone_frozen = False
            print("Backbone unfrozen. Fine-tuning the full network.")

        train_metrics, _, _, _ = run_epoch(
            model,
            loaders["train_loader"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            description=f"Train {epoch}/{config.epochs}",
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics, _, _, _ = run_epoch(
            model,
            loaders["val_loader"],
            criterion,
            device,
            optimizer=None,
            scaler=None,
            description=f"Val {epoch}/{config.epochs}",
        )

        scheduler.step(val_metrics["loss"])

        epoch_history = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_history)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc_ovr']:.4f}"
        )

        val_auc = val_metrics["macro_auc_ovr"]
        if np.isnan(val_auc):
            val_auc = 0.0
        current_score = 0.7 * val_metrics["macro_f1"] + 0.3 * val_auc
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    print(f"Best checkpoint restored from epoch {best_epoch}.")
    return model, history


def evaluate_model(model, loader, class_names, device):
    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred, y_prob = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        description="Test",
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    sample_paths = [path for path, _ in loader.dataset.samples]
    patient_results = aggregate_results_by_patient(
        sample_paths=sample_paths,
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "sample_paths": sample_paths,
        "report_text": report_text,
        "report_dict": report_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "patient_results": patient_results,
    }


def print_dataset_summary(class_names, train_counts, val_counts, test_counts):
    print("Class distribution")
    print("-" * 60)
    print(f"{'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    print("-" * 60)
    for index, class_name in enumerate(class_names):
        print(
            f"{class_name:20s} "
            f"{int(train_counts[index]):8d} "
            f"{int(val_counts[index]):8d} "
            f"{int(test_counts[index]):8d}"
        )
    print("-" * 60)


def print_audit_summary(audit_summary):
    if not audit_summary:
        return

    print("Dataset audit")
    print("-" * 60)
    print(f"Train duplicate groups: {audit_summary['train_duplicate_groups']}")
    print(f"Redundant duplicate train files: {audit_summary['train_redundant_duplicate_files']}")
    print(f"Cross-class duplicate groups: {audit_summary['train_cross_class_duplicate_groups']}")
    print(f"Exact train/val hash overlap after split: {audit_summary['train_val_exact_hash_overlap']}")
    print(f"Exact train/test hash overlap: {audit_summary['train_test_exact_hash_overlap']}")
    print(f"Train samples used after split: {audit_summary['train_samples_after_split']}")
    print(f"Validation samples used after split: {audit_summary['val_samples_after_split']}")
    print("-" * 60)

    patient_summary = audit_summary.get("test_patient_summary")
    if patient_summary:
        print("Test patient-group audit (inferred from filename)")
        print("-" * 60)
        print(
            f"{'Class':20s} {'Images':>8s} {'Patients':>10s} "
            f"{'Repeat':>8s} {'Mean/slc':>10s} {'Max/slc':>8s}"
        )
        print("-" * 60)
        for item in patient_summary:
            print(
                f"{item['class_name']:20s} "
                f"{item['num_images']:8d} "
                f"{item['num_patients']:10d} "
                f"{item['repeated_patient_groups']:8d} "
                f"{item['mean_slices_per_patient']:10.2f} "
                f"{item['max_slices_per_patient']:8d}"
            )
        print("-" * 60)


def plot_dataset_distribution(class_names, train_counts, val_counts, test_counts):
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, train_counts, width=width, label="Train")
    plt.bar(x, val_counts, width=width, label="Validation")
    plt.bar(x + width, test_counts, width=width, label="Test")
    plt.xticks(x, class_names, rotation=20)
    plt.ylabel("Images")
    plt.title("Dataset Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, [item["train_balanced_accuracy"] for item in history], label="Train")
    plt.plot(epochs, [item["val_balanced_accuracy"] for item in history], label="Validation")
    plt.title("Balanced Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, [item["train_macro_f1"] for item in history], label="Train")
    plt.plot(epochs, [item["val_macro_f1"] for item in history], label="Validation")
    plt.title("Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, [item["lr_backbone"] for item in history], label="Backbone LR")
    plt.plot(epochs, [item["lr_head"] for item in history], label="Head LR")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_test_metrics(test_results):
    metrics = test_results["metrics"]
    print("Image-level test metrics")
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro ROC-AUC (OvR): {metrics['macro_auc_ovr']:.4f}")

    patient_results = test_results.get("patient_results")
    if patient_results:
        patient_metrics = patient_results["metrics"]
        print()
        print("Patient-level test metrics (slice probabilities averaged per inferred patient)")
        print(f"Patients evaluated: {patient_results['num_groups']}")
        print(f"Multi-slice patients: {patient_results['num_multi_slice_groups']}")
        print(f"Max slices for one patient: {patient_results['max_group_size']}")
        print(f"Inconsistent label groups: {patient_results['inconsistent_groups']}")
        print(f"Accuracy: {patient_metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {patient_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {patient_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {patient_metrics['weighted_f1']:.4f}")
        print(f"Macro precision: {patient_metrics['macro_precision']:.4f}")
        print(f"Macro recall: {patient_metrics['macro_recall']:.4f}")
        print(f"Macro ROC-AUC (OvR): {patient_metrics['macro_auc_ovr']:.4f}")


def print_classification_report(test_results):
    print("Image-level classification report")
    print(test_results["report_text"])

    patient_results = test_results.get("patient_results")
    if patient_results:
        print("Patient-level classification report")
        print(patient_results["report_text"])


def _select_results_level(test_results, level):
    if level == "patient" and test_results.get("patient_results") is not None:
        return test_results["patient_results"], "Patient-level"
    if level == "image":
        return test_results, "Image-level"
    if level == "patient":
        return test_results, "Image-level"
    raise ValueError("level must be 'patient' or 'image'")


def aggregate_results_by_patient(sample_paths, y_true, y_prob, class_names):
    grouped = {}

    for path, true_label, prob_vector in zip(sample_paths, y_true, y_prob):
        patient_id = _extract_inferred_patient_group_key(path)
        group = grouped.setdefault(
            patient_id,
            {
                "labels": [],
                "probs": [],
            },
        )
        group["labels"].append(int(true_label))
        group["probs"].append(prob_vector)

    patient_ids = []
    grouped_true = []
    grouped_pred = []
    grouped_prob = []
    group_sizes = []
    inconsistent_groups = 0

    for patient_id, group in grouped.items():
        labels = group["labels"]
        label_counts = np.bincount(labels, minlength=len(class_names))
        true_label = int(label_counts.argmax())
        if np.count_nonzero(label_counts) > 1:
            inconsistent_groups += 1

        mean_prob = np.mean(np.vstack(group["probs"]), axis=0)
        pred_label = int(mean_prob.argmax())

        patient_ids.append(patient_id)
        grouped_true.append(true_label)
        grouped_pred.append(pred_label)
        grouped_prob.append(mean_prob)
        group_sizes.append(len(labels))

    grouped_true = np.asarray(grouped_true, dtype=np.int64)
    grouped_pred = np.asarray(grouped_pred, dtype=np.int64)
    grouped_prob = np.asarray(grouped_prob, dtype=np.float32)
    group_sizes = np.asarray(group_sizes, dtype=np.int64)

    metrics = compute_metrics(
        grouped_true,
        grouped_pred,
        grouped_prob,
        len(class_names),
    )
    report_text = classification_report(
        grouped_true,
        grouped_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        grouped_true,
        grouped_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "metrics": metrics,
        "y_true": grouped_true,
        "y_pred": grouped_pred,
        "y_prob": grouped_prob,
        "patient_ids": patient_ids,
        "group_sizes": group_sizes,
        "num_groups": len(patient_ids),
        "num_multi_slice_groups": int(np.sum(group_sizes > 1)),
        "max_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
        "inconsistent_groups": inconsistent_groups,
        "report_text": report_text,
        "report_dict": report_dict,
        "confusion_matrix": confusion_matrix(grouped_true, grouped_pred),
    }


def plot_confusion_matrix_from_results(test_results, class_names, normalize=False, level="patient"):
    selected_results, level_name = _select_results_level(test_results, level)
    cm = selected_results["confusion_matrix"].astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
        fmt = ".2f"
        title = f"{level_name} Normalized Confusion Matrix"
    else:
        fmt = "d"
        cm = cm.astype(int)
        title = f"{level_name} Confusion Matrix"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curves_from_results(test_results, class_names, level="patient"):
    selected_results, level_name = _select_results_level(test_results, level)
    y_true = selected_results["y_true"]
    y_prob = selected_results["y_prob"]
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"{class_name} AUC={roc_auc[i]:.3f}")
        except ValueError:
            roc_auc[i] = float("nan")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{level_name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    macro_auc = float(np.nanmean([roc_auc[i] for i in range(len(class_names))]))
    return {"per_class_auc": roc_auc, "macro_auc": macro_auc, "level": level_name}


def run_experiment(config=None):
    config = config or TrainingConfig()
    seed_everything(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    loaders = prepare_dataloaders(config)
    print_dataset_summary(
        loaders["class_names"],
        loaders["train_counts"],
        loaders["val_counts"],
        loaders["test_counts"],
    )
    print_audit_summary(loaders.get("audit_summary"))

    model = build_model(
        num_classes=len(loaders["class_names"]),
        dropout=config.dropout,
    ).to(device)

    model, history = train_model(model, loaders, config, device)
    test_results = evaluate_model(
        model=model,
        loader=loaders["test_loader"],
        class_names=loaders["class_names"],
        device=device,
    )

    return {
        "config": config,
        "device": str(device),
        "class_names": loaders["class_names"],
        "train_counts": loaders["train_counts"],
        "val_counts": loaders["val_counts"],
        "test_counts": loaders["test_counts"],
        "audit_summary": loaders.get("audit_summary"),
        "model": model,
        "history": history,
        "test_results": test_results,
    }


def refresh_test_evaluation(results):
    config = results["config"]
    device = get_device()
    _, eval_transform = build_transforms(
        config.img_size,
        random_erasing_prob=config.random_erasing_prob,
    )
    test_base_ds = datasets.ImageFolder(config.test_dir)
    class_names = results["class_names"]
    if test_base_ds.classes != class_names:
        raise ValueError("Saved results class order does not match the current test folder.")

    test_ds = TransformedImageFolderSubset(
        samples=test_base_ds.samples,
        classes=class_names,
        class_to_idx=test_base_ds.class_to_idx,
        transform=eval_transform,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    model = results["model"].to(device)
    test_results = evaluate_model(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
    )

    refreshed_results = dict(results)
    refreshed_results["device"] = str(device)
    refreshed_results["class_names"] = class_names
    refreshed_results["test_counts"] = _class_counts(test_ds.targets, len(class_names))
    refreshed_results["test_results"] = test_results
    return refreshed_results
