#!/usr/bin/env python3
"""Draw MOT-format tracking results on image frames.

The script is intentionally read-only for tracker outputs. It searches the
local BoostTrack layout by default and writes visualized frames to
results/img_result/tracking.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[float, float, float, float, float]
FrameResults = Dict[int, List[Tuple[int, Box]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seq", default=None, help="Sequence name, e.g. dancetrack0050.")
    parser.add_argument("--result_txt", type=Path, default=None)
    parser.add_argument("--result_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_ids", default=None, help="Comma-separated frame ids, e.g. 1,25,60.")
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--line_width", type=int, default=3)
    parser.add_argument("--font_scale", type=float, default=0.8)
    parser.add_argument("--make_grid", action="store_true", help="Also save a contact-sheet summary.")
    return parser.parse_args()


def read_mot_txt(path: Path) -> FrameResults:
    frames: FrameResults = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            score = float(parts[6]) if len(parts) > 6 else 1.0
            if w <= 0 or h <= 0:
                continue
            frames[frame].append((tid, (x, y, w, h, score)))
    return frames


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 37) % 180
    sat = 175 + (track_id * 17) % 70
    val = 210 + (track_id * 29) % 45
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def find_dataset_root(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "dancetrack",
        repo_root / "data" / "DanceTrack",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find DanceTrack root under data/dancetrack.")


def find_sequence_dir(dataset_root: Path, split: str, seq: str) -> Path:
    direct = dataset_root / split / seq
    if (direct / "img1").exists():
        return direct
    matches = [p.parent for p in dataset_root.rglob("img1") if p.parent.name == seq]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find sequence {seq!r} with img1 under {dataset_root}.")


def find_result_txt(repo_root: Path, result_dir: Optional[Path], seq: Optional[str]) -> Path:
    if result_dir is not None:
        if seq is None:
            txts = sorted(result_dir.glob("*.txt"))
            if not txts:
                raise FileNotFoundError(f"No txt files under {result_dir}.")
            return txts[0]
        path = result_dir / f"{seq}.txt"
        if path.exists():
            return path
        raise FileNotFoundError(f"Cannot find {path}.")

    search_root = repo_root / "results" / "trackers"
    txts = list(search_root.rglob("*.txt"))
    if seq is not None:
        txts = [p for p in txts if p.stem == seq]
    preferred = []
    for p in txts:
        s = str(p).lower()
        score = 0
        if "memotrack" in s or "phdtrack" in s:
            score += 100
        if "dance" in s:
            score += 20
        if "_post" in s:
            score += 5
        preferred.append((score, p.stat().st_mtime, p))
    if not preferred:
        raise FileNotFoundError("No matching tracking result txt found under results/trackers.")
    return sorted(preferred, reverse=True)[0][2]


def available_images(seq_dir: Path) -> List[Path]:
    imgs = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend((seq_dir / "img1").glob(suffix))
    return sorted(imgs)


def frame_path_for_id(images: Sequence[Path], frame_id: int) -> Path:
    stem = f"{frame_id:08d}"
    for image in images:
        if image.stem == stem:
            return image
    stem = f"{frame_id:06d}"
    for image in images:
        if image.stem == stem:
            return image
    idx = frame_id - 1
    if 0 <= idx < len(images):
        return images[idx]
    raise FileNotFoundError(f"No image found for frame {frame_id}.")


def select_frame_ids(args: argparse.Namespace, frames: FrameResults) -> List[int]:
    available = sorted(frames)
    if args.frame_ids:
        return [int(x) for x in args.frame_ids.split(",") if x.strip()]
    if args.frame_start is not None or args.frame_end is not None:
        start = args.frame_start if args.frame_start is not None else available[0]
        end = args.frame_end if args.frame_end is not None else available[-1]
        return [f for f in available if start <= f <= end][: args.max_frames]
    return available[: args.max_frames]


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], font_scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + baseline + 6), color, -1)
    cv2.putText(img, text, (x + 4, y0 + th + 1), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_frame(
    image: np.ndarray,
    items: Iterable[Tuple[int, Box]],
    line_width: int,
    font_scale: float,
) -> np.ndarray:
    canvas = image.copy()
    for tid, (x, y, w, h, score) in items:
        color = color_for_id(tid)
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(canvas, p1, p2, color, line_width, cv2.LINE_AA)
        draw_label(canvas, f"{tid}", p1[0], p1[1], color, font_scale)
        center = (int(round(x + 0.5 * w)), int(round(y + 0.5 * h)))
        cv2.circle(canvas, center, max(2, line_width), (12, 18, 28), -1, cv2.LINE_AA)
    return canvas


def save_grid(image_paths: Sequence[Path], output_path: Path, cols: int = 4) -> None:
    imgs = [cv2.imread(str(p)) for p in image_paths]
    imgs = [img for img in imgs if img is not None]
    if not imgs:
        return
    thumb_w = 360
    thumbs = []
    for img in imgs:
        h, w = img.shape[:2]
        thumb_h = int(round(thumb_w * h / w))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    max_h = max(t.shape[0] for t in thumbs)
    rows = int(math.ceil(len(thumbs) / cols))
    grid = np.full((rows * max_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * max_h : r * max_h + thumb.shape[0], c * thumb_w : (c + 1) * thumb_w] = thumb
    cv2.imwrite(str(output_path), grid)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    dataset_root = args.dataset_root or find_dataset_root(repo_root)
    result_txt = args.result_txt or find_result_txt(repo_root, args.result_dir, args.seq)
    seq = args.seq or result_txt.stem
    seq_dir = find_sequence_dir(dataset_root, args.split, seq)
    output_dir = args.output_dir or repo_root / "results" / "img_result" / "tracking" / seq
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_mot_txt(result_txt)
    images = available_images(seq_dir)
    selected = select_frame_ids(args, frames)
    written: List[Path] = []
    for frame_id in selected:
        image_path = frame_path_for_id(images, frame_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}.")
        canvas = draw_frame(image, frames.get(frame_id, []), args.line_width, args.font_scale)
        out = output_dir / f"{seq}_{frame_id:06d}.jpg"
        cv2.imwrite(str(out), canvas)
        written.append(out)

    if args.make_grid:
        save_grid(written, output_dir / f"{seq}_summary_grid.jpg")

    print(f"Sequence: {seq}")
    print(f"Result: {result_txt}")
    print(f"Images: {seq_dir / 'img1'}")
    print(f"Written {len(written)} frames to {output_dir}")


if __name__ == "__main__":
    main()
