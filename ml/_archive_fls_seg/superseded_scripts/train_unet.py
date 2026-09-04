"""
U-Net + ResNet34 reference model (segmentation_models_pytorch), matching the
published Watertank baseline (0.7481 mIoU, Singh & Valdenegro-Toro, ICCVW 2021):

    smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=12)

Try encoder_weights loaded from agrija9/ssl-sonar-images (sonar-domain SSL pretrain)
as a documented alternative experiment -- see ml/notebooks/02_baseline_training.ipynb.

This is a REFERENCE model for comparison; the primary pipeline uses YOLOv8-seg.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = ML_DIR / "configs"
CHECKPOINTS_DIR = ML_DIR / "models" / "checkpoints" / "unet"
SPLITS_DIR = ML_DIR / "data" / "splits"

NUM_CLASSES = 12  # Watertank classes (0-10) + net (11)


# ---- Dataset ---------------------------------------------------------------

class SonarSegmentationDataset(Dataset):
    """
    PyTorch dataset for sonar segmentation with YOLO-seg polygon labels.

    Reads grayscale images and reconstructs pixel masks from YOLO polygon labels
    for U-Net training (which expects dense pixel masks, not polygon labels).
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int = 256,
        num_classes: int = NUM_CLASSES,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment

        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        logger.info(
            f"Loaded {len(self.image_paths)} images from {images_dir}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem

        # Load grayscale image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Return a blank sample as fallback
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        h, w = image.shape[:2]

        # Load YOLO polygon labels and reconstruct dense mask
        mask = np.zeros((h, w), dtype=np.int64)
        label_path = self.labels_dir / f"{stem}.txt"

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:  # class + at least 3 points
                        continue

                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # Denormalize polygon points
                    points = []
                    for i in range(0, len(coords) - 1, 2):
                        px = int(coords[i] * w)
                        py = int(coords[i + 1] * h)
                        points.append([px, py])

                    if len(points) >= 3:
                        pts = np.array(points, dtype=np.int32)
                        # Fill polygon on mask (class_id + 1 to reserve 0 for background)
                        cv2.fillPoly(mask, [pts], class_id + 1)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask.astype(np.uint8),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        # Augmentation (simple flips for reference model)
        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # Normalize to [0, 1] and add channel dim
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # (1, H, W)

        return torch.from_numpy(image), torch.from_numpy(mask)


# ---- Loss -------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation."""

    def __init__(self, num_classes: int, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss."""
        pred_soft = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred_soft)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice


# ---- Metrics ----------------------------------------------------------------

def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute mean IoU across all classes."""
    pred_labels = pred.argmax(dim=1)
    ious = []

    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        target_mask = target == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


# ---- Training ---------------------------------------------------------------

def train_unet(
    config_path: Optional[Path] = None,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    image_size: int = 256,
    device: str = "",
):
    """
    Train U-Net + ResNet34 reference model.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        logger.error(
            "segmentation_models_pytorch not installed. Run:\n"
            "  pip install segmentation-models-pytorch"
        )
        sys.exit(1)

    # Load config if provided
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        epochs = cfg.get("training", {}).get("epochs", epochs)
        batch_size = cfg.get("training", {}).get("batch_size", batch_size)
        learning_rate = cfg.get("training", {}).get("learning_rate", learning_rate)
        image_size = cfg.get("training", {}).get("image_size", image_size)

    # Device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Device: {device}")

    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=NUM_CLASSES + 1,  # +1 for background class 0
    ).to(device)

    logger.info(
        f"Model: U-Net + ResNet34, "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Data
    train_ds = SonarSegmentationDataset(
        SPLITS_DIR / "train" / "images",
        SPLITS_DIR / "train" / "labels",
        image_size=image_size,
        augment=True,
    )
    val_ds = SonarSegmentationDataset(
        SPLITS_DIR / "val" / "images",
        SPLITS_DIR / "val" / "labels",
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DiceCELoss(NUM_CLASSES + 1)

    # Training loop
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0
    patience_counter = 0
    patience_limit = 15

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        n_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()

                miou = compute_miou(preds, masks, NUM_CLASSES + 1)
                val_miou += miou
                n_val += 1

        val_loss /= max(n_val, 1)
        val_miou /= max(n_val, 1)

        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            save_path = CHECKPOINTS_DIR / "best_unet_resnet34.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "miou": best_miou,
                },
                save_path,
            )
            logger.info(f"  -> New best model! mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience_limit} epochs)"
                )
                break

    logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
    logger.info(f"Best model saved at: {CHECKPOINTS_DIR / 'best_unet_resnet34.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net + ResNet34 reference segmentation model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "unet_resnet34.yaml",
        help="Config YAML path",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    train_unet(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
