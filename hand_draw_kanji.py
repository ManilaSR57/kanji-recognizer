from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import transforms

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


# ---------------------------------------------------------------------
# CNN architecture: must match the training script checkpoint.
# ---------------------------------------------------------------------

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
        return self.classifier(x)


# ---------------------------------------------------------------------
# Model-compatible preprocessing for the final canvas.
# This mirrors the training-time preprocessing pipeline.
# ---------------------------------------------------------------------

def estimate_background_value(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0

    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]

    border = np.concatenate([top, bottom, left, right])
    return int(np.median(border))


def crop_ink_area_pil(
    img: Image.Image,
    threshold_delta: int,
) -> Tuple[Image.Image, int]:
    """
    Detects ink and crops to its bounding box.

    Works for:
        white strokes on black background
        black strokes on white background
    """
    img = img.convert("L")
    arr = np.array(img)

    bg = estimate_background_value(arr)

    if bg >= 128:
        # Light background, dark ink.
        ink_mask = arr < (bg - threshold_delta)
        fill = 255
    else:
        # Dark background, light ink.
        ink_mask = arr > (bg + threshold_delta)
        fill = 0

    if not ink_mask.any():
        return img, fill

    ys, xs = np.where(ink_mask)

    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())

    return img.crop((left, top, right + 1, bottom + 1)), fill


def pad_to_square_pil(
    img: Image.Image,
    fill: int,
    padding_ratio: float,
) -> Image.Image:
    """
    Pads cropped kanji to a square before resizing.
    """
    img = img.convert("L")

    w, h = img.size
    side = max(w, h)

    extra_pad = int(round(side * padding_ratio))
    canvas_side = max(1, side + 2 * extra_pad)

    canvas = Image.new("L", (canvas_side, canvas_side), fill)

    paste_x = (canvas_side - w) // 2
    paste_y = (canvas_side - h) // 2

    canvas.paste(img, (paste_x, paste_y))
    return canvas


def ensure_output_polarity_pil(
    img: Image.Image,
    current_background_fill: int,
    output_polarity: str,
) -> Tuple[Image.Image, int]:
    """
    output_polarity:
        black_on_white
        white_on_black
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


def preprocess_canvas_for_model(
    canvas_gray: np.ndarray,
    image_size: int,
    output_polarity: str,
    padding_ratio: float,
    crop_delta: int,
) -> Image.Image:
    """
    Final app canvas preprocessing:

        canvas
        -> PIL grayscale image
        -> crop ink area
        -> pad to square
        -> ensure same white/black polarity as trained model
        -> resize to training image size
    """
    img = Image.fromarray(canvas_gray.astype(np.uint8)).convert("L")

    img, bg_fill = crop_ink_area_pil(
        img,
        threshold_delta=crop_delta,
    )

    img = pad_to_square_pil(
        img,
        fill=bg_fill,
        padding_ratio=padding_ratio,
    )

    img, _ = ensure_output_polarity_pil(
        img,
        current_background_fill=bg_fill,
        output_polarity=output_polarity,
    )

    img = img.resize((image_size, image_size), RESAMPLE_LANCZOS)

    return img


class KanjiPredictor:
    def __init__(self, checkpoint_path: Path):
        self.device = self._get_device()
        self.checkpoint_path = checkpoint_path

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

        if "idx_to_kanji" not in checkpoint:
            raise KeyError(
                "Checkpoint must contain 'idx_to_kanji'. "
                "Use the checkpoint saved by train_kanji_cnn.py."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint must contain 'model_state_dict'. "
                "Use the checkpoint saved by train_kanji_cnn.py."
            )

        self.idx_to_kanji = checkpoint["idx_to_kanji"]

        self.image_size = int(checkpoint.get("image_size", 64))
        self.output_polarity = str(
            checkpoint.get("output_polarity", "white_on_black")
        )
        self.padding_ratio = float(checkpoint.get("padding_ratio", 0.12))
        self.crop_delta = int(checkpoint.get("crop_delta", 15))

        self.model = KanjiCNN(
            num_classes=len(self.idx_to_kanji)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @torch.no_grad()
    def predict_canvas(
        self,
        canvas_gray: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        if not np.any(canvas_gray > 0):
            return []

        img = preprocess_canvas_for_model(
            canvas_gray=canvas_gray,
            image_size=self.image_size,
            output_polarity=self.output_polarity,
            padding_ratio=self.padding_ratio,
            crop_delta=self.crop_delta,
        )

        x = self.tensor_transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        k = min(top_k, len(self.idx_to_kanji))

        top_probs, top_indices = probs.topk(k=k, dim=1)

        results: List[Tuple[str, float]] = []

        for prob, idx in zip(top_probs[0], top_indices[0]):
            kanji = self.idx_to_kanji[int(idx)]
            results.append((kanji, float(prob)))

        return results


# ---------------------------------------------------------------------
# Gesture recognition from MediaPipe landmarks.
# ---------------------------------------------------------------------

FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class FingerState:
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_pattern(self) -> Tuple[bool, bool, bool, bool]:
        return self.index, self.middle, self.ring, self.pinky

    def as_text(self) -> str:
        return (
            f"I:{int(self.index)} "
            f"M:{int(self.middle)} "
            f"R:{int(self.ring)} "
            f"P:{int(self.pinky)}"
        )


def landmarks_to_np(hand_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )


def angle_degrees(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Angle ABC in degrees.
    """
    v1 = a - b
    v2 = c - b

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))

    if denom < 1e-8:
        return 0.0

    cos_value = float(np.dot(v1, v2) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))

    return float(np.degrees(np.arccos(cos_value)))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def is_long_finger_extended(
    pts: np.ndarray,
    joints: Tuple[int, int, int, int],
    angle_threshold: float,
    distance_ratio: float,
) -> bool:
    """
    Rule-based long-finger extension test.

    A finger counts as extended when:
        1. MCP-PIP-TIP angle is large.
        2. Tip is farther from wrist than PIP.
        3. Tip is farther from MCP than PIP.
    """
    mcp, pip, _dip, tip = joints
    wrist = pts[0]

    straight_angle = angle_degrees(
        pts[mcp],
        pts[pip],
        pts[tip],
    )

    is_straight = straight_angle >= angle_threshold

    tip_farther_from_wrist = (
        distance(pts[tip], wrist)
        >
        distance(pts[pip], wrist) * distance_ratio
    )

    tip_farther_from_mcp = (
        distance(pts[tip], pts[mcp])
        >
        distance(pts[pip], pts[mcp]) * 1.05
    )

    return bool(
        is_straight
        and tip_farther_from_wrist
        and tip_farther_from_mcp
    )


def classify_gesture(
    hand_landmarks,
    angle_threshold: float,
    distance_ratio: float,
) -> Tuple[str, Optional[FingerState]]:
    if hand_landmarks is None:
        return "NO_HAND", None

    pts = landmarks_to_np(hand_landmarks)

    fingers = FingerState(
        index=is_long_finger_extended(
            pts,
            FINGER_JOINTS["index"],
            angle_threshold,
            distance_ratio,
        ),
        middle=is_long_finger_extended(
            pts,
            FINGER_JOINTS["middle"],
            angle_threshold,
            distance_ratio,
        ),
        ring=is_long_finger_extended(
            pts,
            FINGER_JOINTS["ring"],
            angle_threshold,
            distance_ratio,
        ),
        pinky=is_long_finger_extended(
            pts,
            FINGER_JOINTS["pinky"],
            angle_threshold,
            distance_ratio,
        ),
    )

    pattern = fingers.as_pattern()

    if (pattern == (False, False, False, False) or pattern == (True, False, False, True)):
        return "IDLE", fingers

    if pattern == (True, False, False, False):
        return "DRAW", fingers

    if pattern == (True, True, False, False):
        return "UNDO", fingers

    if pattern == (True, True, True, False):
        return "RECOGNIZE", fingers

    if pattern == (True, True, True, True):
        return "CLEAR", fingers

    return "UNKNOWN", fingers


class GestureDebouncer:
    """
    A raw gesture must stay the same for several frames before it becomes
    the confirmed gesture.

    This prevents accidental command triggering from jitter.
    """
    def __init__(self, stable_frames: int):
        self.stable_frames = stable_frames
        self.last_raw = "UNKNOWN"
        self.raw_count = 0
        self.confirmed = "UNKNOWN"

    def update(self, raw_gesture: str) -> Tuple[str, bool, str]:
        if raw_gesture == self.last_raw:
            self.raw_count += 1
        else:
            self.last_raw = raw_gesture
            self.raw_count = 1

        previous_confirmed = self.confirmed
        changed = False

        if self.raw_count >= self.stable_frames and raw_gesture != self.confirmed:
            self.confirmed = raw_gesture
            changed = True

        return self.confirmed, changed, previous_confirmed


# ---------------------------------------------------------------------
# Drawing canvas.
# ---------------------------------------------------------------------

@dataclass
class SquareROI:
    x: int
    y: int
    size: int

    def contains(self, p: Tuple[int, int]) -> bool:
        px, py = p

        return (
            self.x <= px < self.x + self.size
            and self.y <= py < self.y + self.size
        )

    def map_to_canvas(
        self,
        p: Tuple[int, int],
        canvas_size: int,
    ) -> Tuple[int, int]:
        px, py = p

        cx = int(round((px - self.x) / self.size * (canvas_size - 1)))
        cy = int(round((py - self.y) / self.size * (canvas_size - 1)))

        cx = int(np.clip(cx, 0, canvas_size - 1))
        cy = int(np.clip(cy, 0, canvas_size - 1))

        return cx, cy


def compute_square_roi(
    frame_w: int,
    frame_h: int,
    scale: float,
) -> SquareROI:
    size = int(min(frame_w, frame_h) * scale)
    size = max(32, size)

    x = (frame_w - size) // 2
    y = (frame_h - size) // 2

    return SquareROI(x=x, y=y, size=size)


class PointSmoother:
    """
    Exponential moving average smoother for the fingertip position.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev = None

    def update(self, p: Tuple[int, int]) -> Tuple[int, int]:
        current = np.array(p, dtype=np.float32)

        if self.prev is None:
            self.prev = current
        else:
            self.prev = self.alpha * current + (1.0 - self.alpha) * self.prev

        return int(round(float(self.prev[0]))), int(round(float(self.prev[1])))


class DrawingCanvas:
    def __init__(
        self,
        canvas_size: int,
        brush_size: int,
        min_point_distance: float,
    ):
        self.canvas_size = canvas_size
        self.brush_size = brush_size
        self.min_point_distance = min_point_distance

        # White strokes on black background.
        self.canvas = np.zeros(
            (canvas_size, canvas_size),
            dtype=np.uint8,
        )

        self.strokes: List[List[Tuple[int, int]]] = []
        self.current_stroke: List[Tuple[int, int]] = []

    def start_stroke(self) -> None:
        if self.current_stroke:
            self.finish_stroke()

        self.current_stroke = []

    def add_point(self, p: Tuple[int, int]) -> None:
        p = (int(p[0]), int(p[1]))

        if self.current_stroke:
            last = self.current_stroke[-1]

            if (
                np.hypot(p[0] - last[0], p[1] - last[1])
                <
                self.min_point_distance
            ):
                return

            cv2.line(
                self.canvas,
                last,
                p,
                255,
                self.brush_size,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                self.canvas,
                p,
                max(1, self.brush_size // 2),
                255,
                -1,
                lineType=cv2.LINE_AA,
            )

        self.current_stroke.append(p)

    def finish_stroke(self) -> None:
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []

    def _draw_stroke(
        self,
        stroke: List[Tuple[int, int]],
    ) -> None:
        if not stroke:
            return

        cv2.circle(
            self.canvas,
            stroke[0],
            max(1, self.brush_size // 2),
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

        for p1, p2 in zip(stroke[:-1], stroke[1:]):
            cv2.line(
                self.canvas,
                p1,
                p2,
                255,
                self.brush_size,
                lineType=cv2.LINE_AA,
            )

    def redraw(self) -> None:
        self.canvas.fill(0)

        for stroke in self.strokes:
            self._draw_stroke(stroke)

        if self.current_stroke:
            self._draw_stroke(self.current_stroke)

    def undo(self) -> None:
        self.current_stroke = []

        if self.strokes:
            self.strokes.pop()

        self.redraw()

    def clear(self) -> None:
        self.strokes = []
        self.current_stroke = []
        self.canvas.fill(0)

    def has_ink(self) -> bool:
        return bool(np.any(self.canvas > 0))

    def save(self, path: Path) -> None:
        cv2.imwrite(str(path), self.canvas)


# ---------------------------------------------------------------------
# Display helpers.
# ---------------------------------------------------------------------

def landmark_pixel(
    lm,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int]:
    x = int(np.clip(lm.x, 0.0, 1.0) * (frame_w - 1))
    y = int(np.clip(lm.y, 0.0, 1.0) * (frame_h - 1))

    return x, y


def draw_hand_overlay(
    frame: np.ndarray,
    hand_landmarks,
) -> None:
    h, w = frame.shape[:2]
    points = [landmark_pixel(lm, w, h) for lm in hand_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[a],
            points[b],
            (80, 220, 80),
            2,
            lineType=cv2.LINE_AA,
        )

    for i, p in enumerate(points):
        radius = 5 if i == 8 else 3
        color = (0, 255, 255) if i == 8 else (220, 220, 220)

        cv2.circle(
            frame,
            p,
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )


def put_text_lines(
    image: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_height: int = 24,
) -> None:
    x, y = origin

    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def format_predictions(
    predictions: List[Tuple[str, float]],
) -> List[str]:
    if not predictions:
        return ["Prediction: none"]

    lines = ["Prediction:"]

    for rank, (kanji, prob) in enumerate(predictions, start=1):
        lines.append(f"{rank}. {kanji}  {prob:.3f}")

    return lines


# ---------------------------------------------------------------------
# Main app.
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best_model.pt from train_kanji_cnn.py",
    )

    parser.add_argument(
        "--hand-model",
        type=Path,
        required=True,
        help="Path to MediaPipe hand_landmarker.task",
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=720)

    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--brush-size", type=int, default=14)
    parser.add_argument("--min-point-distance", type=float, default=3.0)
    parser.add_argument("--roi-scale", type=float, default=0.85)
    parser.add_argument("--smooth-alpha", type=float, default=0.35)

    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--command-cooldown-frames", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--finger-angle-threshold", type=float, default=150.0)
    parser.add_argument("--finger-distance-ratio", type=float, default=1.08)

    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.55)

    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("last_air_kanji_canvas.png"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not args.hand_model.exists():
        raise FileNotFoundError(f"MediaPipe hand model not found: {args.hand_model}")

    predictor = KanjiPredictor(args.checkpoint)

    drawer = DrawingCanvas(
        canvas_size=args.canvas_size,
        brush_size=args.brush_size,
        min_point_distance=args.min_point_distance,
    )

    smoother = PointSmoother(alpha=args.smooth_alpha)

    debouncer = GestureDebouncer(
        stable_frames=args.stable_frames,
    )

    predictions: List[Tuple[str, float]] = []

    status_message = "Ready. Use fist to idle, index finger to draw."
    frame_index = 0
    cooldown_until_frame = -1
    last_timestamp_ms = -1

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print("Gesture map:")
    print("  FIST or INDEX + PINKY    -> idle / stop drawing")
    print("  INDEX only               -> draw current stroke")
    print("  INDEX + MIDDLE           -> undo previous stroke")
    print("  INDEX + MIDDLE + RING    -> recognize final canvas")
    print("  FOUR LONG FINGERS        -> clear canvas")
    print("Keyboard fallback: q quit, u undo, c clear, r recognize, s save canvas")
    print(f"Kanji model device: {predictor.device}")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            # Mirror for natural interaction.
            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            roi = compute_square_roi(frame_w, frame_h, args.roi_scale)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(time.monotonic() * 1000)

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            raw_gesture = "NO_HAND"
            finger_state: Optional[FingerState] = None

            hand_landmarks = (
                result.hand_landmarks[0]
                if result.hand_landmarks
                else None
            )

            if hand_landmarks is not None:
                raw_gesture, finger_state = classify_gesture(
                    hand_landmarks,
                    angle_threshold=args.finger_angle_threshold,
                    distance_ratio=args.finger_distance_ratio,
                )

                draw_hand_overlay(frame, hand_landmarks)

            confirmed_gesture, changed, previous_gesture = debouncer.update(
                raw_gesture
            )

            # State transition handling.
            if changed:
                if previous_gesture == "DRAW" and confirmed_gesture != "DRAW":
                    drawer.finish_stroke()
                    smoother.reset()

                if confirmed_gesture == "DRAW":
                    drawer.start_stroke()
                    smoother.reset()
                    status_message = "Drawing stroke."

                command_is_allowed = previous_gesture in {
                    "IDLE",
                    "UNKNOWN",
                    "NO_HAND",
                }

                cooldown_is_over = frame_index >= cooldown_until_frame

                if (
                    confirmed_gesture in {"UNDO", "CLEAR", "RECOGNIZE"}
                    and command_is_allowed
                    and cooldown_is_over
                ):
                    if confirmed_gesture == "UNDO":
                        drawer.undo()
                        predictions = []
                        status_message = "Undid previous stroke."

                    elif confirmed_gesture == "CLEAR":
                        drawer.clear()
                        smoother.reset()
                        predictions = []
                        status_message = "Cleared canvas."

                    elif confirmed_gesture == "RECOGNIZE":
                        drawer.finish_stroke()
                        smoother.reset()

                        predictions = predictor.predict_canvas(
                            drawer.canvas,
                            top_k=args.top_k,
                        )

                        if predictions:
                            joined = ", ".join(
                                [f"{k}:{p:.3f}" for k, p in predictions]
                            )

                            status_message = f"Recognized: {joined}"
                            print(status_message)
                        else:
                            status_message = "Canvas is empty; nothing to recognize."

                    cooldown_until_frame = (
                        frame_index + args.command_cooldown_frames
                    )

            # Drawing update.
            # This happens only after DRAW is stable.
            if confirmed_gesture == "DRAW" and hand_landmarks is not None:
                fingertip = landmark_pixel(
                    hand_landmarks[8],
                    frame_w,
                    frame_h,
                )

                cv2.circle(
                    frame,
                    fingertip,
                    9,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

                if roi.contains(fingertip):
                    canvas_point = roi.map_to_canvas(
                        fingertip,
                        drawer.canvas_size,
                    )

                    canvas_point = smoother.update(canvas_point)
                    drawer.add_point(canvas_point)
                else:
                    # Leaving the drawing box breaks the stroke instead of
                    # drawing a long line across the canvas edge.
                    if drawer.current_stroke:
                        drawer.finish_stroke()

                    smoother.reset()
                    status_message = "Index finger is outside the drawing box."

            # Draw the square drawing region on the camera view.
            cv2.rectangle(
                frame,
                (roi.x, roi.y),
                (roi.x + roi.size, roi.y + roi.size),
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            finger_text = (
                finger_state.as_text()
                if finger_state is not None
                else "I:? M:? R:? P:?"
            )

            put_text_lines(
                frame,
                [
                    f"Raw: {raw_gesture} | Confirmed: {confirmed_gesture}",
                    f"Fingers: {finger_text}",
                    f"Strokes: {len(drawer.strokes) + int(bool(drawer.current_stroke))}",
                    status_message[:80],
                ],
                origin=(12, 28),
                color=(255, 255, 255),
            )

            # Prepare canvas display panel.
            canvas_bgr = cv2.cvtColor(drawer.canvas, cv2.COLOR_GRAY2BGR)

            display_h = frame_h

            canvas_display = cv2.resize(
                canvas_bgr,
                (display_h, display_h),
                interpolation=cv2.INTER_NEAREST,
            )

            put_text_lines(
                canvas_display,
                format_predictions(predictions),
                origin=(12, 32),
                color=(180, 255, 180),
                scale=0.7,
                thickness=2,
                line_height=30,
            )

            help_lines = [
                "Gestures:",
                "fist or (index & pinky)=idle",
                "index=draw",
                "2 fingers=undo",
                "3 fingers=recognize",
                "4 fingers=clear",
            ]

            put_text_lines(
                canvas_display,
                help_lines,
                origin=(12, display_h - 150),
                color=(200, 200, 200),
                scale=0.52,
                thickness=1,
                line_height=22,
            )

            combined = np.hstack([frame, canvas_display])

            cv2.imshow("Air Kanji Recognition", combined)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("u"):
                drawer.undo()
                predictions = []
                status_message = "Keyboard: undo."

            elif key == ord("c"):
                drawer.clear()
                smoother.reset()
                predictions = []
                status_message = "Keyboard: clear."

            elif key == ord("r"):
                drawer.finish_stroke()
                smoother.reset()

                predictions = predictor.predict_canvas(
                    drawer.canvas,
                    top_k=args.top_k,
                )

                status_message = "Keyboard: recognize."
                print(format_predictions(predictions))

            elif key == ord("s"):
                drawer.finish_stroke()
                drawer.save(args.save_path)

                status_message = f"Saved canvas to {args.save_path}"
                print(status_message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()