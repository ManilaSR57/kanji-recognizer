from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------
# CSV and label handling
# ------------------------------------------------------------

def parse_kanji_label(value: object) -> str:
    """
    Accepts labels like:
        日
        U+65E5
        65E5
        0x65E5
        \\u65e5
        26085     # decimal Unicode code point

    Returns the actual kanji character whenever possible.
    """
    s = str(value).strip()

    if s == "" or s.lower() == "nan":
        raise ValueError("empty label")

    # Already a single character, e.g. 日
    if len(s) == 1:
        return s

    # Python-style escaped string, e.g. "\\u65e5"
    if s.startswith("\\u") or s.startswith("\\U"):
        try:
            decoded = s.encode("utf-8").decode("unicode_escape")
            if decoded:
                return decoded
        except UnicodeDecodeError:
            pass

    upper = s.upper().replace(" ", "")

    # U+65E5
    if upper.startswith("U+"):
        return chr(int(upper[2:], 16))

    # 0x65E5
    if upper.startswith("0X"):
        return chr(int(upper[2:], 16))

    # 65E5
    if re.fullmatch(r"[0-9A-F]{4,6}", upper) and any(c in "ABCDEF" for c in upper):
        return chr(int(upper, 16))

    # Decimal Unicode code point, e.g. 26085
    if re.fullmatch(r"\d+", s):
        codepoint = int(s)
        if 0 <= codepoint <= 0x10FFFF:
            return chr(codepoint)

    # Fallback: keep the string as-is.
    return s


def resolve_image_path(data_dir: Path, image_id: object) -> Optional[Path]:
    """
    Your CSV first column corresponds to file names like 000000.png.

    This function tries several possibilities:
        000000 -> 000000.png
        0      -> 000000.png
        123    -> 000123.png
        123    -> 123.png
        000123.png -> 000123.png
    """
    raw = str(image_id).strip()

    if raw == "" or raw.lower() == "nan":
        return None

    if raw.lower().endswith(".png"):
        stem = raw[:-4]
    else:
        stem = raw

    candidates: List[Path] = []

    # Exact match.
    candidates.append(data_dir / f"{stem}.png")

    # Numeric variants.
    if re.fullmatch(r"\d+(\.0+)?", stem):
        n = int(float(stem))
        candidates.append(data_dir / f"{n:06d}.png")
        candidates.append(data_dir / f"{n}.png")

    # Remove duplicate candidates while preserving order.
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            unique_candidates.append(p)
            seen.add(p)

    for p in unique_candidates:
        if p.exists():
            return p

    return None


def load_records(data_dir: Path, csv_name: str) -> List[dict]:
    csv_path = data_dir / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)

    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns in {csv_path}. "
            f"Got columns: {list(df.columns)}"
        )

    image_id_col = df.columns[0]
    label_col = df.columns[1]

    records: List[dict] = []
    missing_images = 0
    bad_labels = 0

    for _, row in df.iterrows():
        image_id = row[image_id_col]
        raw_label = row[label_col]

        image_path = resolve_image_path(data_dir, image_id)

        if image_path is None:
            missing_images += 1
            continue

        try:
            kanji = parse_kanji_label(raw_label)
        except Exception:
            bad_labels += 1
            continue

        records.append({
            "image_id": str(image_id),
            "path": str(image_path),
            "kanji": kanji,
        })

    if len(records) == 0:
        raise RuntimeError(
            "No usable image-label records were found. "
            "Check the CSV columns and file names."
        )

    print(f"Loaded usable records: {len(records)}")
    print(f"Missing images skipped: {missing_images}")
    print(f"Bad labels skipped: {bad_labels}")

    return records


def filter_rare_classes(records: List[dict], min_samples_per_class: int) -> List[dict]:
    counts = Counter(r["kanji"] for r in records)

    kept = [
        r for r in records
        if counts[r["kanji"]] >= min_samples_per_class
    ]

    removed_classes = [
        label for label, count in counts.items()
        if count < min_samples_per_class
    ]

    if removed_classes:
        print(
            f"Removed {len(removed_classes)} classes with fewer than "
            f"{min_samples_per_class} samples."
        )

    return kept


def make_label_maps(records: List[dict]) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(set(r["kanji"] for r in records))
    kanji_to_idx = {kanji: idx for idx, kanji in enumerate(classes)}
    idx_to_kanji = classes
    return kanji_to_idx, idx_to_kanji


def stratified_split(
    records: List[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Splits separately inside each kanji class.

    This is better than a simple random split because every class should be
    represented in train/validation/test, assuming enough samples exist.
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for r in records:
        by_label[r["kanji"]].append(r)

    train_records = []
    val_records = []
    test_records = []

    for label, items in by_label.items():
        items = items[:]
        rng.shuffle(items)

        n = len(items)

        if n == 1:
            train_records.extend(items)
            continue

        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        if test_ratio > 0 and n >= 3:
            n_test = max(1, n_test)
        else:
            n_test = 0

        if val_ratio > 0 and n - n_test >= 2:
            n_val = max(1, n_val)
        else:
            n_val = 0

        # Always keep at least one sample for training.
        while n - n_val - n_test < 1:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break

        test_records.extend(items[:n_test])
        val_records.extend(items[n_test:n_test + n_val])
        train_records.extend(items[n_test + n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return train_records, val_records, test_records


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

def composite_alpha_on_background(img: Image.Image, fill: int = 255) -> Image.Image:
    """
    Handles transparent PNGs.

    If the PNG has alpha, we place it on a solid background before converting
    to grayscale. For typical black handwriting on transparent background,
    white fill=255 is appropriate.
    """
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (fill, fill, fill, 255))
        out = Image.alpha_composite(bg, rgba)
        return out.convert("L")

    return img.convert("L")


def estimate_background_value(arr: np.ndarray) -> int:
    """
    Estimate the background from border pixels.

    This lets us handle both:
        black ink on white background
        white ink on black background
    """
    if arr.size == 0:
        return 255

    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]

    border = np.concatenate([top, bottom, left, right])
    return int(np.median(border))


def crop_ink_area(
    img: Image.Image,
    threshold_delta: int = 15,
) -> Tuple[Image.Image, int]:
    """
    Crop to the bounding box of the written character.

    Returns:
        cropped_image
        background_fill_value: 255 if background is white-ish, 0 if black-ish
    """
    img = img.convert("L")
    arr = np.array(img)

    bg = estimate_background_value(arr)

    if bg >= 128:
        # White/light background, dark ink.
        ink_mask = arr < (bg - threshold_delta)
        fill = 255
    else:
        # Black/dark background, light ink.
        ink_mask = arr > (bg + threshold_delta)
        fill = 0

    if not ink_mask.any():
        return img, fill

    ys, xs = np.where(ink_mask)

    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())

    cropped = img.crop((left, top, right + 1, bottom + 1))
    return cropped, fill


def pad_to_square(
    img: Image.Image,
    fill: int,
    padding_ratio: float = 0.12,
) -> Image.Image:
    """
    Pad the cropped kanji to a square canvas.

    This prevents distortion when resizing to 64x64 or 96x96.
    """
    img = img.convert("L")

    w, h = img.size
    side = max(w, h)

    extra_pad = int(round(side * padding_ratio))
    canvas_side = side + 2 * extra_pad

    canvas = Image.new("L", (canvas_side, canvas_side), fill)

    paste_x = (canvas_side - w) // 2
    paste_y = (canvas_side - h) // 2

    canvas.paste(img, (paste_x, paste_y))
    return canvas


def ensure_output_polarity(
    img: Image.Image,
    current_background_fill: int,
    output_polarity: str,
) -> Tuple[Image.Image, int]:
    """
    output_polarity:
        black_on_white: black strokes, white background
        white_on_black: white strokes, black background
    """
    if output_polarity == "black_on_white":
        desired_bg = 255
    elif output_polarity == "white_on_black":
        desired_bg = 0
    else:
        raise ValueError(f"Unknown output_polarity: {output_polarity}")

    if current_background_fill == desired_bg:
        return img, desired_bg

    return ImageOps.invert(img), desired_bg


def preprocess_kanji_image(
    image_path: str | Path,
    image_size: int,
    output_polarity: str,
    padding_ratio: float,
    crop_delta: int,
    alpha_background: int,
) -> Tuple[Image.Image, int]:
    """
    Full preprocessing pipeline:

        PNG
        -> handle alpha
        -> grayscale
        -> crop ink area
        -> pad to square
        -> optional inversion
        -> resize

    Returns:
        processed PIL image
        background fill value after polarity conversion
    """
    img = Image.open(image_path)

    img = composite_alpha_on_background(img, fill=alpha_background)
    img, bg_fill = crop_ink_area(img, threshold_delta=crop_delta)
    img = pad_to_square(img, fill=bg_fill, padding_ratio=padding_ratio)
    img, output_bg_fill = ensure_output_polarity(
        img,
        current_background_fill=bg_fill,
        output_polarity=output_polarity,
    )

    img = img.resize((image_size, image_size), RESAMPLE_LANCZOS)

    return img, output_bg_fill


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class KanjiDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        kanji_to_idx: Dict[str, int],
        image_size: int,
        output_polarity: str,
        padding_ratio: float,
        crop_delta: int,
        alpha_background: int,
        augment: bool,
    ):
        self.records = records
        self.kanji_to_idx = kanji_to_idx
        self.image_size = image_size
        self.output_polarity = output_polarity
        self.padding_ratio = padding_ratio
        self.crop_delta = crop_delta
        self.alpha_background = alpha_background
        self.augment = augment

        # After preprocessing, background is known.
        if output_polarity == "black_on_white":
            affine_fill = 255
        else:
            affine_fill = 0

        if augment:
            # Do NOT use flips for kanji.
            # Small rotations/translations/scale changes simulate handwriting variation.
            self.augmentation = transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
                fill=affine_fill,
            )
        else:
            self.augmentation = None

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]

        img, _ = preprocess_kanji_image(
            image_path=record["path"],
            image_size=self.image_size,
            output_polarity=self.output_polarity,
            padding_ratio=self.padding_ratio,
            crop_delta=self.crop_delta,
            alpha_background=self.alpha_background,
        )

        if self.augmentation is not None:
            img = self.augmentation(img)

        x = self.to_tensor(img)
        y = self.kanji_to_idx[record["kanji"]]

        return x, torch.tensor(y, dtype=torch.long)


# ------------------------------------------------------------
# CNN model
# ------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KanjiCNN(nn.Module):
    """
    Simple CNN baseline.

    Input:
        [batch_size, 1, image_size, image_size]

    Output:
        [batch_size, num_classes]
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.10),
            ConvBlock(64, 128, dropout=0.15),
            ConvBlock(128, 256, dropout=0.20),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits


# ------------------------------------------------------------
# Training / evaluation helpers
# ------------------------------------------------------------

@torch.no_grad()
def topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[int, int]:
    num_classes = logits.shape[1]
    max_k = min(max(topk), num_classes)

    _, pred = logits.topk(max_k, dim=1)

    result = {}

    for k in topk:
        kk = min(k, num_classes)
        correct = pred[:, :kk].eq(targets.view(-1, 1)).any(dim=1).sum().item()
        result[k] = int(correct)

    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0

    progress = tqdm(loader, desc="train", leave=False)

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        correct = topk_correct(logits.detach(), y, topk=(1, 5))
        total_top1 += correct[1]
        total_top5 += correct[5]

        progress.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "top1": f"{total_top1 / total_samples:.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0

    progress = tqdm(loader, desc=desc, leave=False)

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        correct = topk_correct(logits, y, topk=(1, 5))
        total_top1 += correct[1]
        total_top5 += correct[5]

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    kanji_to_idx: Dict[str, int],
    idx_to_kanji: List[str],
    args: argparse.Namespace,
    epoch: int,
    val_metrics: Dict[str, float],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "kanji_to_idx": kanji_to_idx,
        "idx_to_kanji": idx_to_kanji,
        "image_size": args.image_size,
        "output_polarity": args.output_polarity,
        "padding_ratio": args.padding_ratio,
        "crop_delta": args.crop_delta,
        "alpha_background": args.alpha_background,
        "epoch": epoch,
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint, path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="kanji_codes.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/kanji_cnn"))

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--output-polarity",
        type=str,
        choices=["black_on_white", "white_on_black"],
        default="black_on_white",
        help=(
            "The normalized image polarity used by the model. "
            "Use black_on_white for black strokes on white background. "
            "Use white_on_black for white strokes on black background."
        ),
    )

    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=2,
        help="Classes with fewer samples than this are removed.",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.12,
        help="Extra padding around cropped ink before resizing.",
    )

    parser.add_argument(
        "--crop-delta",
        type=int,
        default=15,
        help="Difference from background needed for a pixel to count as ink.",
    )

    parser.add_argument(
        "--alpha-background",
        type=int,
        default=255,
        help="Background used when compositing transparent PNGs. 255 = white.",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # -----------------------------
    # Load metadata
    # -----------------------------

    records = load_records(args.data_dir, args.csv_name)
    records = filter_rare_classes(records, args.min_samples_per_class)

    if len(records) == 0:
        raise RuntimeError("No records remain after filtering rare classes.")

    kanji_to_idx, idx_to_kanji = make_label_maps(records)
    num_classes = len(idx_to_kanji)

    print(f"Number of classes: {num_classes}")
    print(f"Number of images: {len(records)}")

    save_json(kanji_to_idx, args.out_dir / "kanji_to_idx.json")
    save_json(idx_to_kanji, args.out_dir / "idx_to_kanji.json")

    # -----------------------------
    # Split dataset
    # -----------------------------

    train_records, val_records, test_records = stratified_split(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if len(val_records) == 0:
        raise RuntimeError(
            "Validation split is empty. "
            "Use a larger dataset or lower --min-samples-per-class."
        )

    print(f"Train images: {len(train_records)}")
    print(f"Val images:   {len(val_records)}")
    print(f"Test images:  {len(test_records)}")

    pd.DataFrame(train_records).to_csv(args.out_dir / "train_split.csv", index=False)
    pd.DataFrame(val_records).to_csv(args.out_dir / "val_split.csv", index=False)
    pd.DataFrame(test_records).to_csv(args.out_dir / "test_split.csv", index=False)

    # -----------------------------
    # Create datasets and loaders
    # -----------------------------

    train_dataset = KanjiDataset(
        records=train_records,
        kanji_to_idx=kanji_to_idx,
        image_size=args.image_size,
        output_polarity=args.output_polarity,
        padding_ratio=args.padding_ratio,
        crop_delta=args.crop_delta,
        alpha_background=args.alpha_background,
        augment=not args.no_augment,
    )

    val_dataset = KanjiDataset(
        records=val_records,
        kanji_to_idx=kanji_to_idx,
        image_size=args.image_size,
        output_polarity=args.output_polarity,
        padding_ratio=args.padding_ratio,
        crop_delta=args.crop_delta,
        alpha_background=args.alpha_background,
        augment=False,
    )

    test_dataset = KanjiDataset(
        records=test_records,
        kanji_to_idx=kanji_to_idx,
        image_size=args.image_size,
        output_polarity=args.output_polarity,
        padding_ratio=args.padding_ratio,
        crop_delta=args.crop_delta,
        alpha_background=args.alpha_background,
        augment=False,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # -----------------------------
    # Create model
    # -----------------------------

    model = KanjiCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # -----------------------------
    # Train
    # -----------------------------

    best_val_top1 = -1.0
    history = []

    best_path = args.out_dir / "best_model.pt"
    last_path = args.out_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="val",
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)

        print(
            f"train loss={train_metrics['loss']:.4f}, "
            f"top1={train_metrics['top1']:.4f}, "
            f"top5={train_metrics['top5']:.4f}"
        )

        print(
            f"val   loss={val_metrics['loss']:.4f}, "
            f"top1={val_metrics['top1']:.4f}, "
            f"top5={val_metrics['top5']:.4f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            kanji_to_idx=kanji_to_idx,
            idx_to_kanji=idx_to_kanji,
            args=args,
            epoch=epoch,
            val_metrics=val_metrics,
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]

            save_checkpoint(
                path=best_path,
                model=model,
                kanji_to_idx=kanji_to_idx,
                idx_to_kanji=idx_to_kanji,
                args=args,
                epoch=epoch,
                val_metrics=val_metrics,
            )

            print(f"Saved new best model to: {best_path}")

    # -----------------------------
    # Final test evaluation
    # -----------------------------

    if len(test_dataset) > 0:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            desc="test",
        )

        print("\nBest model test result:")
        print(
            f"test loss={test_metrics['loss']:.4f}, "
            f"top1={test_metrics['top1']:.4f}, "
            f"top5={test_metrics['top5']:.4f}"
        )

        save_json(test_metrics, args.out_dir / "test_metrics.json")

    print("\nDone.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()