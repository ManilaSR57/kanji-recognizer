from pathlib import Path
import argparse
import torch

from train_kanji_cnn import (
    KanjiCNN,
    preprocess_kanji_image,
)
from torchvision import transforms


def predict_one(
    checkpoint_path: Path,
    image_path: Path,
    top_k: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    idx_to_kanji = checkpoint["idx_to_kanji"]
    num_classes = len(idx_to_kanji)

    model = KanjiCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img, _ = preprocess_kanji_image(
        image_path=image_path,
        image_size=checkpoint["image_size"],
        output_polarity=checkpoint["output_polarity"],
        padding_ratio=checkpoint["padding_ratio"],
        crop_delta=checkpoint["crop_delta"],
        alpha_background=checkpoint["alpha_background"],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(k=top_k, dim=1)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        kanji = idx_to_kanji[int(idx)]
        results.append((kanji, float(prob)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    results = predict_one(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        top_k=args.top_k,
    )

    for rank, (kanji, prob) in enumerate(results, start=1):
        print(f"{rank}. {kanji}  probability={prob:.4f}")


if __name__ == "__main__":
    main()