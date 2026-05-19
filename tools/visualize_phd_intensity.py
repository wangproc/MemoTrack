import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import dataset
from default_settings import GeneralSettings, canonical_dataset_name, get_detector_path_and_im_size
from external.adaptors import detector
from memotracker.memo_tracker import MemoTracker
from memotracker.PHD_dataset_config import PHDDatasetConfig


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


def _unwrap_bgr_image(np_img):
    if isinstance(np_img, np.ndarray):
        return np_img.copy()
    if hasattr(np_img, "numpy"):
        return np_img.numpy().copy()
    if isinstance(np_img, (list, tuple)) and np_img:
        return _unwrap_bgr_image(np_img[0])
    raise TypeError(f"Unsupported image container type: {type(np_img)!r}")


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


def _project_spatial_heatmap(phd_filter, image_shape, grid_stride=4, max_components=32, min_weight=1e-4, cov_scale=1.0):
    height, width = image_shape[:2]
    weights = np.asarray(phd_filter.w_update, dtype=float)
    if weights.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    order = np.argsort(weights)[::-1]
    keep = [int(idx) for idx in order if weights[int(idx)] > min_weight][:max_components]
    if not keep:
        return np.zeros((height, width), dtype=np.float32)

    xs = np.arange(0, width, grid_stride, dtype=np.float32)
    ys = np.arange(0, height, grid_stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heat = np.zeros_like(grid_x, dtype=np.float64)

    H = phd_filter.model["H"]
    R = phd_filter.model["R"]

    for idx in keep:
        mu = H @ phd_filter.m_update[:, idx]
        cov = H @ phd_filter.P_update[:, :, idx] @ H.T + R
        cov_xy = (cov[:2, :2] * max(float(cov_scale), 1e-6)).astype(np.float64)
        cov_xy += np.eye(2, dtype=np.float64) * 1e-6
        det_cov = float(np.linalg.det(cov_xy))
        if det_cov <= 0:
            continue

        try:
            inv_cov = np.linalg.inv(cov_xy)
        except np.linalg.LinAlgError:
            continue

        dx = grid_x - float(mu[0])
        dy = grid_y - float(mu[1])
        exponent = -0.5 * (
            inv_cov[0, 0] * dx * dx
            + 2.0 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy * dy
        )
        exponent = np.clip(exponent, -60.0, 0.0)
        pdf = np.exp(exponent) / (2.0 * np.pi * np.sqrt(det_cov))
        heat += float(weights[idx]) * pdf

    blur_sigma = 1.25 + 0.95 * np.sqrt(max(float(cov_scale), 1e-6))
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    heat = np.maximum(heat, 0.0)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = np.maximum(heat, 0.0)
    return heat.astype(np.float32)


def _normalize_heatmap(heatmap):
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)
    positive = heatmap[heatmap > 0]
    if positive.size == 0:
        return np.zeros_like(heatmap, dtype=np.float32)

    low = max(float(np.percentile(positive, 68)), 1e-12)
    high = max(float(np.percentile(positive, 99.5)), low * 1.01)
    norm = ((heatmap - low) / max(high - low, 1e-12)).astype(np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.power(norm, 0.90).astype(np.float32)
    norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=1.8, sigmaY=1.8)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _build_blue_tinted_background(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bg = np.zeros_like(frame_bgr, dtype=np.float32)
    bg[..., 0] = 34.0 + 122.0 * gray
    bg[..., 1] = 12.0 + 56.0 * gray
    bg[..., 2] = 8.0 + 24.0 * gray
    return np.clip(bg, 0, 255)


def _colorize_heatmap(heatmap_norm, background_floor=0.08):
    heat_vis = np.clip(background_floor + (1.0 - background_floor) * np.clip(heatmap_norm, 0.0, 1.0), 0.0, 1.0)
    return cv2.applyColorMap(np.uint8(heat_vis * 255.0), cv2.COLORMAP_JET)


def _overlay_heatmap(frame_bgr, heatmap_norm):
    color = _colorize_heatmap(heatmap_norm).astype(np.float32)
    faint_bg = _build_blue_tinted_background(frame_bgr)
    alpha = np.clip(0.68 + 0.18 * np.power(np.clip(heatmap_norm, 0.0, 1.0), 0.75), 0.0, 0.90).astype(np.float32)
    blended = faint_bg * (1.0 - alpha[..., None]) + color * alpha[..., None]
    blended = 0.88 * blended + 0.12 * frame_bgr.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def get_args():
    parser = argparse.ArgumentParser(description="Render PHD posterior intensity overlays for one sequence or all sequences in the chosen split.")
    parser.add_argument("--dataset", type=str, default="dance")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--sequence", type=str, default="", help="Sequence name, or comma-separated sequence names. Leave empty to process all sequences in the split.")
    parser.add_argument("--annotation_file", type=str, default="", help="Optional annotation json override, e.g. val_half.json or train.json.")
    parser.add_argument("--split_name", type=str, default="", help="Optional image split override, e.g. train or val.")
    parser.add_argument("--frame_id", type=int, default=0, help="If > 0, render only this frame id in the selected sequence.")
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--end_frame", type=int, default=0, help="0 means no upper limit.")
    parser.add_argument("--save_stride", type=int, default=1)
    parser.add_argument("--result_root", type=str, default="results/img_result")
    parser.add_argument("--exp_name", type=str, default="MemoTrack_heatmap")
    parser.add_argument("--detector_path", type=str, default="")
    parser.add_argument("--reid_path", type=str, default="")
    parser.add_argument("--grid_stride", type=int, default=4)
    parser.add_argument("--max_components", type=int, default=32)
    parser.add_argument("--min_component_weight", type=float, default=1e-4)
    parser.add_argument("--cov_scale", type=float, default=2.0)
    parser.add_argument("--save_frame", action="store_true", help="Also save the raw frame.")
    parser.add_argument("--save_heatmap", action="store_true", help="Also save the standalone colorized heatmap.")
    parser.add_argument("--jpg_quality", type=int, default=92)
    parser.add_argument("--cfg", nargs="*", default=None, help="Optional key=value tracker overrides.")
    args = parser.parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    if args.reid_path:
        os.environ["MEMOTRACK_REID_PATH"] = args.reid_path
        os.environ["BOOSTTRACK_REID_PATH"] = args.reid_path
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

    tracker = None
    current_video = None
    saved = 0

    def should_save(frame_id):
        if args.frame_id > 0:
            return frame_id == args.frame_id
        if frame_id < args.start_frame:
            return False
        if args.end_frame > 0 and frame_id > args.end_frame:
            return False
        return (frame_id - args.start_frame) % max(args.save_stride, 1) == 0

    for (img, np_img), _, info, _ in loader:
        frame_id = int(info[2].item())
        file_name = info[4][0]
        video_name = file_name.split("/")[0]
        absolute_frame = _absolute_frame_id(file_name, frame_id)
        if args.sequence and args.frame_id > 0 and video_name == args.sequence and frame_id > args.frame_id:
            break

        img = img.cuda()
        tag = f"{video_name}:{frame_id}"

        if frame_id == 1 or video_name != current_video:
            if tracker is not None:
                tracker.dump_cache()
                det.dump_cache()
            tracker = MemoTracker(video_name=video_name, cfg_override=cfg_override)
            current_video = video_name
            print(f"Rendering PHD overlays for {video_name}")

        pred = det(img, tag)
        if pred is None:
            continue

        frame_bgr = _unwrap_bgr_image(np_img[0])
        tracker.update(pred, img, frame_bgr, tag)

        if should_save(frame_id):
            out_dir = Path(args.result_root) / "phd_intensity" / video_name
            out_dir.mkdir(parents=True, exist_ok=True)
            heat_raw = _project_spatial_heatmap(
                tracker.phd,
                frame_bgr.shape,
                grid_stride=args.grid_stride,
                max_components=args.max_components,
                min_weight=args.min_component_weight,
                cov_scale=args.cov_scale,
            )
            heat_norm = _normalize_heatmap(heat_raw)
            overlay = _overlay_heatmap(frame_bgr, heat_norm)

            base_name = f"{video_name}_{absolute_frame:06d}"
            overlay_path = out_dir / f"{base_name}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.jpg_quality, 70, 100))])

            if args.save_frame:
                frame_path = out_dir / f"{base_name}_frame.jpg"
                cv2.imwrite(str(frame_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.jpg_quality, 70, 100))])
            if args.save_heatmap:
                heat_path = out_dir / f"{base_name}_heatmap.jpg"
                cv2.imwrite(
                    str(heat_path),
                    _colorize_heatmap(heat_norm),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.jpg_quality, 70, 100))],
                )

            saved += 1
            if saved <= 3 or saved % 200 == 0:
                print(f"Saved PHD overlay {saved}: {overlay_path}")

            if args.frame_id > 0:
                break

    det.dump_cache()
    if tracker is not None:
        tracker.dump_cache()

    if saved == 0:
        target = f"frame {args.frame_id} for sequence {args.sequence}" if args.frame_id > 0 else "the requested selection"
        raise RuntimeError(f"No PHD overlays were rendered for {target}.")
    print(f"Saved {saved} PHD overlay image(s) under {Path(args.result_root) / 'phd_intensity'}")


if __name__ == "__main__":
    main()
