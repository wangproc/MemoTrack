import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GT-aligned side-by-side comparison images for MemoTrack and ByteTrack core.")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", choices=("mot17", "dance", "all"), default="all")
    parser.add_argument("--max_frames_per_seq", type=int, default=0, help="0 means all frames.")
    return parser.parse_args()


def read_mot_txt(path: Path):
    frames = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
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


def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=float)


def iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)
    b1 = np.asarray(boxes1, dtype=float)
    b2 = np.asarray(boxes2, dtype=float)
    xx1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    yy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    xx2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    yy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def match_boxes(gt_boxes, pred_boxes, iou_thr):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    ious = iou_matrix(gt_boxes, pred_boxes)
    rows, cols = linear_sum_assignment(1.0 - ious)
    tp = 0
    matched_gt = set()
    matched_pred = set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        if ious[r, c] >= iou_thr:
            tp += 1
            matched_gt.add(r)
            matched_pred.add(c)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def color_for_identity(identity):
    hue = (int(identity) * 37) % 180
    sat = 180 + (int(identity) * 17) % 60
    val = 210 + (int(identity) * 29) % 45
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_labeled_boxes(image, tlwhs, labels, colors):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(height, width) / 320)))
    font_scale = max(0.45, min(height, width) / 1700.0)
    font_thickness = max(1, thickness - 1)

    for tlwh, label, color in zip(tlwhs, labels, colors):
        x1, y1, w, h = tlwh
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x1 + w))
        y2 = int(round(y1 + h))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        if not label:
            continue
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


def add_title_bar(image, text):
    bar_h = 48
    canvas = np.full((image.shape[0] + bar_h, image.shape[1], 3), 250, dtype=np.uint8)
    canvas[bar_h:] = image
    cv2.putText(canvas, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, lineType=cv2.LINE_AA)
    return canvas


def load_annotation_frames(annotation_path: Path):
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_info = {}
    seq_frames = defaultdict(dict)
    for img in data["images"]:
        file_name = img["file_name"]
        seq = file_name.split("/")[0]
        absolute_frame = int(Path(file_name).stem)
        result_frame = int(img["frame_id"])
        image_info[img["id"]] = {
            "seq": seq,
            "result_frame": result_frame,
            "absolute_frame": absolute_frame,
            "file_name": file_name,
        }
        seq_frames[seq][absolute_frame] = {
            "result_frame": result_frame,
            "file_name": file_name,
            "gt_records": [],
        }

    for ann in data["annotations"]:
        info = image_info.get(int(ann["image_id"]))
        if info is None:
            continue
        x, y, w, h = ann["bbox"]
        seq_frames[info["seq"]][info["absolute_frame"]]["gt_records"].append({
            "track_id": int(ann["track_id"]),
            "bbox": np.array([x, y, x + w, y + h], dtype=float),
        })

    return seq_frames


def assign_gt_display(pred_items, gt_records, iou_thr):
    tlwhs = [box for _, box in pred_items]
    if not tlwhs:
        return [], [], []

    pred_boxes = [tlwh_to_xyxy(box) for box in tlwhs]
    gt_boxes = [item["bbox"] for item in gt_records]
    labels = []
    colors = []
    matched_pred_to_gt = {}

    if gt_boxes:
        ious = iou_matrix(gt_boxes, pred_boxes)
        rows, cols = linear_sum_assignment(1.0 - ious)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if ious[r, c] >= iou_thr:
                matched_pred_to_gt[c] = gt_records[r]["track_id"]

    for idx in range(len(tlwhs)):
        gt_id = matched_pred_to_gt.get(idx, None)
        if gt_id is None:
            labels.append("fp")
            colors.append((150, 150, 150))
        else:
            labels.append(str(int(gt_id)))
            colors.append(color_for_identity(int(gt_id)))
    return tlwhs, labels, colors


def get_dataset_config(repo_root: Path, dataset: str):
    if dataset == "mot17":
        return {
            "annotation_path": repo_root / "data" / "MOT17" / "annotations" / "val_half.json",
            "image_root": repo_root / "data" / "MOT17" / "train",
            "ours_result_dir": repo_root / "results" / "trackers" / "MOT17-val" / "MemoTrack" / "data",
            "byte_result_dir": repo_root / "results" / "trackers" / "MOT17-val" / "ByteCore_MOT17_val_raw" / "data",
            "output_root": repo_root / "results" / "img_result" / "aligned_all_comparisons" / "mot17",
            "ours_name": "MemoTrack",
            "byte_name": "ByteTrack-Core",
        }
    if dataset == "dance":
        return {
            "annotation_path": repo_root / "data" / "dancetrack" / "annotations" / "val.json",
            "image_root": repo_root / "data" / "dancetrack" / "val",
            "ours_result_dir": repo_root / "results" / "img_result" / "qualitative" / "data",
            "byte_result_dir": repo_root / "results" / "trackers" / "DANCE-val" / "ByteCore_DANCE_val_raw" / "data",
            "output_root": repo_root / "results" / "img_result" / "aligned_all_comparisons" / "dance",
            "ours_name": "MemoTrack",
            "byte_name": "ByteTrack-Core",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def process_dataset(repo_root: Path, dataset: str, max_frames_per_seq: int):
    cfg = get_dataset_config(repo_root, dataset)
    seq_frames = load_annotation_frames(cfg["annotation_path"])
    output_root = cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    seq_names = sorted(
        seq for seq in seq_frames
        if (cfg["ours_result_dir"] / f"{seq}.txt").exists() and (cfg["byte_result_dir"] / f"{seq}.txt").exists()
    )
    if not seq_names:
        raise RuntimeError(f"No overlapping result files found for dataset={dataset}.")

    summary_rows = []
    total_written = 0
    for seq in seq_names:
        ours_frames = read_mot_txt(cfg["ours_result_dir"] / f"{seq}.txt")
        byte_frames = read_mot_txt(cfg["byte_result_dir"] / f"{seq}.txt")
        seq_output = output_root / seq
        seq_output.mkdir(parents=True, exist_ok=True)

        frame_items = sorted(seq_frames[seq].items())
        if max_frames_per_seq > 0:
            frame_items = frame_items[:max_frames_per_seq]

        for idx, (absolute_frame, meta) in enumerate(frame_items, start=1):
            result_frame = meta["result_frame"]
            file_name = meta["file_name"]
            gt_records = meta["gt_records"]

            image_path = cfg["image_root"] / file_name
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            ours_items = ours_frames.get(result_frame, [])
            byte_items = byte_frames.get(result_frame, [])

            ours_tlwhs, ours_labels, ours_colors = assign_gt_display(ours_items, gt_records, 0.5)
            byte_tlwhs, byte_labels, byte_colors = assign_gt_display(byte_items, gt_records, 0.5)

            ours_img = draw_labeled_boxes(frame, ours_tlwhs, ours_labels, ours_colors)
            byte_img = draw_labeled_boxes(frame, byte_tlwhs, byte_labels, byte_colors)

            gt_boxes = [item["bbox"] for item in gt_records]
            ours_boxes = [tlwh_to_xyxy(box) for box in ours_tlwhs]
            byte_boxes = [tlwh_to_xyxy(box) for box in byte_tlwhs]
            ours_tp, ours_fp, ours_fn = match_boxes(gt_boxes, ours_boxes, 0.5)
            byte_tp, byte_fp, byte_fn = match_boxes(gt_boxes, byte_boxes, 0.5)

            ours_title = f"{cfg['ours_name']}  TP {ours_tp}  FP {ours_fp}  FN {ours_fn}"
            byte_title = f"{cfg['byte_name']}  TP {byte_tp}  FP {byte_fp}  FN {byte_fn}"
            ours_panel = add_title_bar(ours_img, ours_title)
            byte_panel = add_title_bar(byte_img, byte_title)

            spacer = np.full((ours_panel.shape[0], 18, 3), 255, dtype=np.uint8)
            merged = np.concatenate([ours_panel, spacer, byte_panel], axis=1)
            footer_h = 42
            canvas = np.full((merged.shape[0] + footer_h, merged.shape[1], 3), 248, dtype=np.uint8)
            canvas[:merged.shape[0]] = merged
            footer = (
                f"{seq}  frame {absolute_frame:06d}  "
                f"GT {len(gt_boxes)}  "
                f"adv {(byte_fp + byte_fn) - (ours_fp + ours_fn)}"
            )
            cv2.putText(canvas, footer, (14, merged.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)

            out_name = f"{seq}_{absolute_frame:06d}_compare.jpg"
            out_path = seq_output / out_name
            cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            summary_rows.append({
                "dataset": dataset,
                "seq": seq,
                "absolute_frame": absolute_frame,
                "result_frame": result_frame,
                "file_name": file_name,
                "output_path": str(out_path),
                "gt_count": len(gt_boxes),
                "ours_tp": ours_tp,
                "ours_fp": ours_fp,
                "ours_fn": ours_fn,
                "byte_tp": byte_tp,
                "byte_fp": byte_fp,
                "byte_fn": byte_fn,
                "advantage": (byte_fp + byte_fn) - (ours_fp + ours_fn),
            })

            total_written += 1
            if total_written <= 3 or total_written % 500 == 0:
                print(f"[{dataset}] saved {total_written}: {out_path}")

    csv_path = output_root / "comparison_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "dataset", "seq", "absolute_frame", "result_frame", "file_name", "output_path",
            "gt_count", "ours_tp", "ours_fp", "ours_fn", "byte_tp", "byte_fp", "byte_fn", "advantage",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[{dataset}] wrote {total_written} comparison image(s) under {output_root}")


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    datasets = ["mot17", "dance"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(repo_root, dataset, args.max_frames_per_seq)


if __name__ == "__main__":
    main()
