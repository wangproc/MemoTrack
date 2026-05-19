import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import dataset
import utils
from default_settings import GeneralSettings, canonical_dataset_name, get_detector_path_and_im_size
from external.adaptors import detector
from memotracker.memo_tracker import MemoTracker
from memotracker.PHD_dataset_config import PHDDatasetConfig
from visualize_phd_intensity import _normalize_heatmap, _overlay_heatmap, _project_spatial_heatmap


def _parse_cfg_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --cfg item: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        low = value.strip().lower()
        if low == "true":
            parsed = True
        elif low == "false":
            parsed = False
        else:
            try:
                parsed = float(value) if any(c in value for c in ".eE") else int(value)
            except ValueError:
                parsed = value
        overrides[key.strip()] = parsed
    return overrides


def _parse_sequence_list(sequence_arg):
    if not sequence_arg:
        return None
    seqs = [item.strip() for item in sequence_arg.split(",") if item.strip()]
    return seqs or None


def _absolute_frame_id(file_name, fallback_frame_id):
    stem = Path(file_name).stem
    try:
        return int(stem)
    except ValueError:
        return int(fallback_frame_id)


def _unwrap_bgr_image(np_img):
    if isinstance(np_img, np.ndarray):
        return np_img.copy()
    if hasattr(np_img, "numpy"):
        return np_img.numpy().copy()
    if isinstance(np_img, (list, tuple)) and np_img:
        return _unwrap_bgr_image(np_img[0])
    raise TypeError(f"Unsupported image container type: {type(np_img)!r}")


def _color_for_track_id(track_id):
    hue = (int(track_id) * 0.618033988749895) % 1.0
    sat = 0.75
    val = 0.98
    hsv = np.uint8([[[int(hue * 179), int(sat * 255), int(val * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_track_boxes(image, tlwhs, ids):
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
        color = _color_for_track_id(track_id)

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


def get_args():
    parser = argparse.ArgumentParser(description="Render MemoTrack tracking results for one or more sequences.")
    parser.add_argument("--dataset", type=str, default="dance")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--sequence", type=str, default="", help="Sequence name, or comma-separated sequence names. Leave empty to process all sequences in the split.")
    parser.add_argument("--annotation_file", type=str, default="", help="Optional annotation json override, e.g. train.json for full MOT17 train sequences.")
    parser.add_argument("--split_name", type=str, default="", help="Optional image split override, e.g. train or val.")
    parser.add_argument("--save_start", type=int, default=1)
    parser.add_argument("--save_end", type=int, default=0, help="0 means no upper limit.")
    parser.add_argument("--save_stride", type=int, default=1)
    parser.add_argument("--result_root", type=str, default="results/img_result")
    parser.add_argument("--exp_name", type=str, default="MemoTrack_qualitative")
    parser.add_argument("--detector_path", type=str, default="")
    parser.add_argument("--reid_path", type=str, default="")
    parser.add_argument("--jpg_quality", type=int, default=92)
    parser.add_argument("--save_phd_overlay", action="store_true", help="Save blue-background PHD overlays in the same pass.")
    parser.add_argument("--phd_grid_stride", type=int, default=4)
    parser.add_argument("--phd_max_components", type=int, default=32)
    parser.add_argument("--phd_min_component_weight", type=float, default=1e-4)
    parser.add_argument("--phd_cov_scale", type=float, default=2.0)
    parser.add_argument("--cfg", nargs="*", default=None, help="Optional key=value tracker overrides.")
    args = parser.parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    if args.reid_path:
        os.environ["MEMOTRACK_REID_PATH"] = args.reid_path
        os.environ["BOOSTTRACK_REID_PATH"] = args.reid_path
    if args.save_end > 0 and args.save_end < args.save_start:
        raise ValueError("--save_end must be greater than or equal to --save_start.")
    return args


def main():
    args = get_args()
    GeneralSettings.values["dataset"] = args.dataset
    GeneralSettings.values["test_dataset"] = False
    GeneralSettings.values["use_embedding"] = True
    GeneralSettings.values["use_ecc"] = True
    PHDDatasetConfig.apply_general(args.dataset)

    cfg_override = _parse_cfg_overrides(args.cfg)
    detector_path, size = get_detector_path_and_im_size(args)
    det = detector.Detector("yolox", detector_path, args.dataset)
    seq_names = _parse_sequence_list(args.sequence)
    loader = dataset.get_mot_loader(
        args.dataset,
        False,
        data_dir=args.data_dir,
        size=size,
        seq_names=seq_names,
        annotation_file=args.annotation_file or None,
        split_name=args.split_name or None,
    )

    data_dir = Path(args.result_root) / "qualitative" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tracker = None
    current_video = None
    current_results = []
    saved_count = 0
    jpg_quality = int(np.clip(args.jpg_quality, 70, 100))

    def flush_results(video_name):
        if not video_name:
            return
        result_path = data_dir / f"{video_name}.txt"
        utils.write_results_no_score(str(result_path), current_results)
        print(f"Saved MOT-format tracking result to {result_path}")

    for (img, np_img), _, info, _ in loader:
        frame_id = int(info[2].item())
        file_name = info[4][0]
        video_name = file_name.split("/")[0]
        absolute_frame = _absolute_frame_id(file_name, frame_id)

        img = img.cuda()
        tag = f"{video_name}:{frame_id}"

        if frame_id == 1 or video_name != current_video:
            if tracker is not None:
                flush_results(current_video)
                tracker.dump_cache()
                det.dump_cache()
            tracker = MemoTracker(video_name=video_name, cfg_override=cfg_override)
            current_video = video_name
            current_results = []
            print(f"Rendering tracking results for {video_name}")

        pred = det(img, tag)
        if pred is None:
            continue

        frame_bgr = _unwrap_bgr_image(np_img[0])
        targets = tracker.update(pred, img, frame_bgr, tag)
        tlwhs, ids, confs = utils.filter_targets(
            targets,
            GeneralSettings["aspect_ratio_thresh"],
            GeneralSettings["min_box_area"],
        )
        current_results.append((frame_id, tlwhs, ids, confs))

        in_range = frame_id >= args.save_start and (args.save_end <= 0 or frame_id <= args.save_end)
        if in_range and (frame_id - args.save_start) % max(args.save_stride, 1) == 0:
            render_dir = Path(args.result_root) / "qualitative" / video_name
            render_dir.mkdir(parents=True, exist_ok=True)
            vis = _draw_track_boxes(frame_bgr, tlwhs, ids)
            out_path = render_dir / f"{video_name}_{absolute_frame:06d}.jpg"
            cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            saved_count += 1
            if saved_count <= 3 or saved_count % 200 == 0:
                print(f"Saved qualitative frame {saved_count}: {out_path}")

            if args.save_phd_overlay:
                phd_dir = Path(args.result_root) / "phd_intensity" / video_name
                phd_dir.mkdir(parents=True, exist_ok=True)
                heat_raw = _project_spatial_heatmap(
                    tracker.phd,
                    frame_bgr.shape,
                    grid_stride=args.phd_grid_stride,
                    max_components=args.phd_max_components,
                    min_weight=args.phd_min_component_weight,
                    cov_scale=args.phd_cov_scale,
                )
                heat_norm = _normalize_heatmap(heat_raw)
                phd_overlay = _overlay_heatmap(frame_bgr, heat_norm)
                phd_path = phd_dir / f"{video_name}_{absolute_frame:06d}_overlay.jpg"
                cv2.imwrite(str(phd_path), phd_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

    det.dump_cache()
    if tracker is not None:
        flush_results(current_video)
        tracker.dump_cache()
    print(f"Saved {saved_count} rendered tracking frame(s) under {Path(args.result_root) / 'qualitative'}")
    if args.save_phd_overlay:
        print(f"Saved matching PHD overlays under {Path(args.result_root) / 'phd_intensity'}")


if __name__ == "__main__":
    main()
