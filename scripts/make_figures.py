"""
Generate the figures used in README.md.

    python scripts/make_figures.py

Outputs:
    assets/training_curves.png   loss / top-1 / top-5 per epoch (from runs/history.csv)
    assets/predictions.png       sample drawing -> model input -> top-5 prediction
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import font_manager
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_kanji_cnn import KanjiCNN, preprocess_kanji_image  # noqa: E402

ASSETS = ROOT / "assets"
CHECKPOINT = ROOT / "runs" / "best_model.pt"
HISTORY = ROOT / "runs" / "history.csv"
SAMPLES = sorted((ROOT / "samples").glob("*.png"))

TRAIN_COLOR = "#4C72B0"
VAL_COLOR = "#DD8452"


def use_cjk_font() -> None:
    """Pick an installed font that can render kanji, if there is one."""
    candidates = ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "MS Gothic", "Droid Sans Fallback"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            return


def plot_training_curves() -> Path:
    df = pd.read_csv(HISTORY)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], color=TRAIN_COLOR, label="train")
    ax.plot(df["epoch"], df["val_loss"], color=VAL_COLOR, label="validation")
    ax.set_title("Loss (cross-entropy, label smoothing 0.05)")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_top1"] * 100, color=TRAIN_COLOR, label="train top-1")
    ax.plot(df["epoch"], df["val_top1"] * 100, color=VAL_COLOR, label="validation top-1")
    ax.plot(df["epoch"], df["val_top5"] * 100, color=VAL_COLOR, linestyle="--", label="validation top-5")
    ax.set_title("Accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    out = ASSETS / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_predictions(top_k: int = 5) -> Path:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    idx_to_kanji = checkpoint["idx_to_kanji"]

    model = KanjiCNN(num_classes=len(idx_to_kanji))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(11, 3.2 * len(SAMPLES)),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]}, squeeze=False)

    for row, path in zip(axes, SAMPLES):
        img, _ = preprocess_kanji_image(
            image_path=path,
            image_size=checkpoint["image_size"],
            output_polarity=checkpoint["output_polarity"],
            padding_ratio=checkpoint["padding_ratio"],
            crop_delta=checkpoint["crop_delta"],
            alpha_background=checkpoint["alpha_background"],
        )

        with torch.no_grad():
            probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
        top_probs, top_idx = probs.topk(top_k)
        labels = [idx_to_kanji[int(i)] for i in top_idx]

        row[0].imshow(plt.imread(path), cmap="gray")
        row[0].set_title(f"Input drawing ({path.name})")
        row[1].imshow(img, cmap="gray")
        row[1].set_title(f"Model input ({img.width}x{img.height})")
        for ax in row[:2]:
            ax.axis("off")

        ax = row[2]
        y = list(range(top_k))[::-1]
        colors = [VAL_COLOR] + [TRAIN_COLOR] * (top_k - 1)
        ax.barh(y, top_probs.numpy() * 100, color=colors)
        ax.set_yticks(y, labels, fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_xlabel("probability (%)")
        ax.set_title(f"Top-{top_k} prediction")
        ax.spines[["top", "right"]].set_visible(False)
        for yi, p in zip(y, top_probs.numpy() * 100):
            ax.text(p + 1, yi, f"{p:.1f}%", va="center")

    fig.tight_layout()
    out = ASSETS / "predictions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    use_cjk_font()
    print(plot_training_curves())
    print(plot_predictions())


if __name__ == "__main__":
    main()
