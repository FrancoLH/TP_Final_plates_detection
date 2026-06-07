from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PATCH_SIZE = 640
_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_patch_dataset(
    split_dir: str | Path,
    patch_size: int = PATCH_SIZE,
    neg_per_pos: int = 1,
    max_neg_attempts: int = 20,
) -> tuple[list[Image.Image], list[int]]:
    """Build positive/negative patch lists from a YOLO-format split directory.

    Positives: cropped annotated plate regions resized to patch_size×patch_size.
    Negatives: random crops that do not overlap any annotation.
    """
    split_dir = Path(split_dir)
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    patches: list[Image.Image] = []
    labels: list[int] = []

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
        if not lines:
            continue

        stem = lbl_path.stem
        img_path = next(
            (
                img_dir / f"{stem}{ext}"
                for ext in (".jpg", ".png")
                if (img_dir / f"{stem}{ext}").exists()
            ),
            None,
        )
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Parse all annotations as pixel boxes
        gt_boxes: list[tuple[int, int, int, int]] = []
        for line in lines:
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                gt_boxes.append((x1, y1, x2, y2))
                patches.append(img.crop((x1, y1, x2, y2)).resize((patch_size, patch_size)))
                labels.append(1)

        # Sample negatives
        crop_w = max(patch_size, int(np.mean([b[2] - b[0] for b in gt_boxes])))
        crop_h = max(patch_size, int(np.mean([b[3] - b[1] for b in gt_boxes])))

        for _ in range(neg_per_pos * len(gt_boxes)):
            for _ in range(max_neg_attempts):
                if W <= crop_w or H <= crop_h:
                    break
                rx1 = random.randint(0, W - crop_w)
                ry1 = random.randint(0, H - crop_h)
                rx2, ry2 = rx1 + crop_w, ry1 + crop_h
                candidate = (rx1, ry1, rx2, ry2)
                if all(_iou(candidate, gt) < 0.1 for gt in gt_boxes):
                    patches.append(img.crop(candidate).resize((patch_size, patch_size)))
                    labels.append(0)
                    break

    return patches, labels


class PlateDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        patches: list[Image.Image],
        labels: list[int],
        transform: Any = None,
    ) -> None:
        self.patches = patches
        self.labels = labels
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.transform(self.patches[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def train_model(
    model: nn.Module,
    train_ldr: DataLoader,  # type: ignore[type-arg]
    val_ldr: DataLoader,  # type: ignore[type-arg]
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
    model_path: str = "best_model.pth",
) -> dict[str, list]:
    device = next(model.parameters()).device
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    best_vloss = float("inf")
    patience_count = 0
    history: dict[str, list] = {
        k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy", "auc", "val_auc")
    }

    for epoch in range(epochs):
        model.train()
        t_loss: float = 0.0
        t_probs: list[float] = []
        t_labels: list[float] = []
        for X_b, y_b in train_ldr:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b).view(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_b)
            t_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            t_labels.extend(y_b.cpu().numpy())

        model.eval()
        v_loss: float = 0.0
        v_probs: list[float] = []
        v_labels: list[float] = []
        with torch.no_grad():
            for X_b, y_b in val_ldr:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b).view(-1)
                v_loss += criterion(logits, y_b).item() * len(y_b)
                v_probs.extend(torch.sigmoid(logits).cpu().numpy())
                v_labels.extend(y_b.cpu().numpy())

        t_loss /= len(train_ldr.dataset)  # type: ignore[arg-type]
        v_loss /= len(val_ldr.dataset)  # type: ignore[arg-type]
        t_acc = ((np.array(t_probs) >= 0.5) == np.array(t_labels)).mean()
        v_acc = ((np.array(v_probs) >= 0.5) == np.array(v_labels)).mean()
        t_auc = roc_auc_score(t_labels, t_probs)
        v_auc = roc_auc_score(v_labels, v_probs)

        for k, val in zip(history, (t_loss, v_loss, t_acc, v_acc, t_auc, v_auc)):
            history[k].append(val)

        scheduler.step(v_loss)

        if v_loss < best_vloss:
            best_vloss = v_loss
            patience_count = 0
            torch.save(copy.deepcopy(model.state_dict()), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Ep {epoch + 1:3d}  loss={t_loss:.4f} val={v_loss:.4f}  "
            f"acc={t_acc:.4f} val_acc={v_acc:.4f}  auc={t_auc:.4f} val_auc={v_auc:.4f}"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    return history


def plot_history(history: dict[str, list], title: str = "Learning curves") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)
    for ax, (metric, label) in zip(
        axes, [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC-ROC")]
    ):
        ax.plot(history[metric], label="Train", linewidth=2)
        ax.plot(history[f"val_{metric}"], label="Val", linewidth=2, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    y_true: np.ndarray,
    model_name: str = "Model",
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    all_probs: list[float] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for X_b, _ in loader:
            logits = model(X_b.to(device)).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))

    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {model_name} — Test Set Evaluation")
    print(f"{sep}")
    print(classification_report(y_true, y_pred, target_names=["No plate", "Plate"]))
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No plate", "Plate"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.show()
    return y_prob, y_pred
