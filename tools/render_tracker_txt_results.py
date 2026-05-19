import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from default_settings import canonical_dataset_name


def parse_args():
    parser = argparse.ArgumentParser(description="Render MOT-format tracking txt files onto source images.")
    parser.add_argument("--dataset", type=str, default="mot20")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--annotation_file", type=str, default="")
    parser.add_argument("--split_name", type=str, default="")
    parser.add_argument("--sequence", type=str, default="", help="Sequence name, or comma-separated sequence names.")
    parser.add_argument("--result_dir", type=Path, required=True, help="Directory containing MOT-format txt results.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory for rendered jpg files.")
    parser.add_argument("--jpg_quality", type=int, default=92)
    parser.add_argument("--random_sequence", action="store_true", help="Randomly render one eligible sequence only.")
    parser.add_argument("--samples_per_sequence", type=int, default=0, help="Randomly sample this many non-empty frames per sequence.")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--flat_output", action="store_true", help="Write all rendered frames directly under output_root.")
    return parser.parse_args()


def parse_sequence_list(sequence_arg):
    if not sequence_arg:
        return None
    seqs = [item.strip() for item in sequence_arg.split(",") if item.strip()]
    return set(seqs) or None


def dataset_layout(dataset, annotation_file, split_name):
    dataset = canonical_dataset_name(dataset)
    if dataset == "mot17":
        return "MOT17", annotation_file or "val_half.json", split_name or "train"
    if dataset == "mot20":
        return "MOT20", annotation_file or "val_half.json", split_name or "train"
    if dataset == "dance":
        return "dancetrack", annotation_file or "val.json", split_name or "val"
    if dataset == "sportsmot":
        return "sportsmot_publish/dataset", annotation_file or "test.json", split_name or "test"
    raise ValueError(f"Unsupported dataset: {dataset}")


def read_mot_txt(path):
    frames = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            if w <= 0 or h <= 0:
                continue
            frames[frame].append((track_id, np.array([x, y, w, h], dtype=float)))
    return frames


def load_annotation_frames(annotation_path):
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    seq_frames = defaultdict(dict)
    for img in data["images"]:
        file_name = img["file_name"]
        seq = file_name.split("/")[0]
        absolute_frame = int(Path(file_name).stem)
        result_frame = int(img["frame_id"])
        seq_frames[seq][absolute_frame] = {
            "result_frame": result_frame,
            "file_name": file_name,
        }
    return seq_frames


def color_for_track_id(track_id):
    hue = (int(track_id) * 0.618033988749895) % 1.0
    sat = 0.75
    val = 0.98
    hsv = np.uint8([[[int(hue * 179), int(sat * 255), int(val * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_track_boxes(image, tlwhs, ids):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(height, width) / 320)))
    font_scale = max(0.45, min(height, width) / 1700.0)
    font_thickness = max(1, thickness - 1)

    for tlwh, track_id in zip(tlwhs, ids):
        x1, y1, w, h = tlwh
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x1 + w))
        y2 = int(round(y1 + h))
        color = color_for_track_id(track_id)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        label = str(int(track_id))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_pad = max(2, thickness)
        bg_x1 = x1
        bg_y2 = max(th + baseline + 2 * text_pad, y1 + 1)
        bg_y1 = max(0, y1 - (th + baseline + 2 * text_pad))
        bg_x2 = min(width - 1, x1 + tw + 2 * text_pad)
        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1, lineType=cv2.LINE_AA)
        text_org = (bg_x1 + text_pad, bg_y2 - baseline - text_pad)
        cv2.putText(
            canvas,
            label,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_root_name, annotation_file, split_name = dataset_layout(args.dataset, args.annotation_file, args.split_name)
    annotation_path = args.data_dir / dataset_root_name / "annotations" / annotation_file
    image_root = args.data_dir / dataset_root_name / split_name
    seq_filter = parse_sequence_list(args.sequence)
    seq_frames = load_annotation_frames(annotation_path)

    result_dir = args.result_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jpg_quality = int(np.clip(args.jpg_quality, 70, 100))

    txt_paths = sorted(result_dir.glob("*.txt"))
    if args.random_sequence:
        rng.shuffle(txt_paths)

    written = 0
    for txt_path in txt_paths:
        seq = txt_path.stem
        if seq_filter and seq not in seq_filter:
            continue
        if seq not in seq_frames:
            continue

        frames = read_mot_txt(txt_path)
        seq_output = output_root if args.flat_output else output_root / seq
        seq_output.mkdir(parents=True, exist_ok=True)

        seq_items = sorted(seq_frames[seq].items())
        if args.samples_per_sequence > 0:
            non_empty_items = [item for item in seq_items if frames.get(item[1]["result_frame"])]
            sample_pool = non_empty_items or seq_items
            sample_count = min(args.samples_per_sequence, len(sample_pool))
            seq_items = sorted(rng.sample(sample_pool, sample_count))

        seq_written = 0
        for absolute_frame, meta in seq_items:
            image_path = image_root / meta["file_name"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            frame_items = frames.get(meta["result_frame"], [])
            tlwhs = [box for _, box in frame_items]
            ids = [track_id for track_id, _ in frame_items]
            rendered = draw_track_boxes(frame, tlwhs, ids)
            out_path = seq_output / f"{seq}_{absolute_frame:06d}.jpg"
            cv2.imwrite(str(out_path), rendered, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

            written += 1
            seq_written += 1
            if written <= 3 or written % 200 == 0:
                print(f"Rendered {written}: {out_path}")

        print(f"Sequence {seq}: saved {seq_written} frame(s) to {seq_output}")
        if args.random_sequence and seq_written > 0:
            break

    print(f"Saved {written} rendered frame(s) under {output_root}")


if __name__ == "__main__":
    main()
