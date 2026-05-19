import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path

import dataset
import utils
from default_settings import GeneralSettings, canonical_dataset_name, get_detector_path_and_im_size
from external.adaptors import detector
from memotracker.GBI import GBInterpolation
from memotracker.PHD_dataset_config import PHDDatasetConfig
from memotracker.memo_tracker import MemoTracker


def _parse_cfg_overrides(items):
    overrides = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'Invalid --cfg item: {item}. Expected key=value.')
        key, value = item.split('=', 1)
        low = value.strip().lower()
        if low == 'true':
            parsed = True
        elif low == 'false':
            parsed = False
        else:
            try:
                parsed = float(value) if any(c in value for c in '.eE') else int(value)
            except ValueError:
                parsed = value
        overrides[key.strip()] = parsed
    return overrides


def _flat_zip(input_dir, output_zip):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Submission input directory not found: {input_dir}')
    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        raise RuntimeError(f'No txt files found in {input_dir}')
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in txt_files:
            zf.write(txt_file, arcname=txt_file.name)
    print(f'Created submission zip {output_zip} with {len(txt_files)} txt files.')


def get_main_args():
    parser = argparse.ArgumentParser(description='Run MemoTrack for online multi-object tracking.')
    parser.add_argument('--dataset', type=str, default='mot17')
    parser.add_argument('--result_folder', type=str, default='results/trackers/')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_dataset', action='store_true')
    parser.add_argument('--all_mot17_detectors', action='store_true', help='include DPM/SDP for MOT17 instead of FRCNN only')
    parser.add_argument('--video_filter', type=str, default='', help='comma-separated sequence names to run')
    parser.add_argument('--exp_name', type=str, default='MemoTrack')
    parser.add_argument('--detector_path', type=str, default='')
    parser.add_argument('--reid_path', type=str, default='')
    parser.add_argument('--cfg', nargs='*', default=None, help='optional key=value parameter overrides')
    parser.add_argument('--post_mode', choices=['auto', 'post', 'post_gbi', 'both', 'none'], default='auto')
    parser.add_argument('--make_submission', action='store_true', help='create a flat zip submission after tracking')
    parser.add_argument('--submission_zip', type=str, default='', help='optional output zip path')
    parser.add_argument('--no_post', action='store_true', help='do not run post-processing')
    args = parser.parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    if args.reid_path:
        os.environ['MEMOTRACK_REID_PATH'] = args.reid_path
        os.environ['BOOSTTRACK_REID_PATH'] = args.reid_path  # Backward-compatible for cached legacy scripts.

    if args.dataset == 'mot17':
        args.result_folder = os.path.join(args.result_folder, 'MOT17-val')
    elif args.dataset == 'mot20':
        args.result_folder = os.path.join(args.result_folder, 'MOT20-val')
    elif args.dataset == 'dance':
        args.result_folder = os.path.join(args.result_folder, 'DANCE-val')
    elif args.dataset == 'sportsmot':
        args.result_folder = os.path.join(args.result_folder, 'SPORTSMOT-val')

    if args.test_dataset:
        args.result_folder = args.result_folder.replace('-val', '-test')
    return args


def main():
    args = get_main_args()
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['test_dataset'] = args.test_dataset
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = True
    PHDDatasetConfig.apply_general(args.dataset)
    cfg_override = _parse_cfg_overrides(args.cfg)
    video_filter = {v.strip() for v in args.video_filter.split(',') if v.strip()}

    detector_path, size = get_detector_path_and_im_size(args)
    det = detector.Detector('yolox', detector_path, args.dataset)
    loader = dataset.get_mot_loader(
        args.dataset,
        args.test_dataset,
        data_dir=args.data_dir,
        size=size,
        seq_names=sorted(video_filter) if video_filter else None,
    )

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0.0

    for (img, np_img), label, info, idx in loader:
        frame_id = info[2].item()
        video_name = info[4][0].split('/')[0]
        if video_filter and video_name not in video_filter:
            continue
        if args.dataset == 'mot17' and not args.all_mot17_detectors and 'FRCNN' not in video_name:
            continue
        tag = f'{video_name}:{frame_id}'
        if video_name not in results:
            results[video_name] = []

        img = img.cuda()
        print(f'Processing {video_name}:{frame_id}\r', end='')

        if frame_id == 1:
            print(f'Initializing tracker for {video_name}')
            print(f'Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}')
            if tracker is not None:
                tracker.dump_cache()
            tracker = MemoTracker(video_name=video_name, cfg_override=cfg_override)

        pred = det(img, tag)
        start_time = time.time()
        if pred is None:
            continue

        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids, confs = utils.filter_targets(
            targets,
            GeneralSettings['aspect_ratio_thresh'],
            GeneralSettings['min_box_area'],
        )

        total_time += time.time() - start_time
        frame_count += 1
        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f'Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}')
    det.dump_cache()
    if tracker is not None:
        tracker.dump_cache()

    folder = os.path.join(args.result_folder, args.exp_name, 'data')
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f'{name}.txt')
        utils.write_results_no_score(result_filename, res)
    print(f'Finished, results saved to {folder}')

    default_post = PHDDatasetConfig.default_submission_post(args.dataset)
    effective_post_mode = default_post if args.post_mode == 'auto' else args.post_mode

    if not args.no_post and effective_post_mode != 'none':
        post_folder = os.path.join(args.result_folder, args.exp_name + '_post')
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, 'data')
        interval = 1000
        utils.dti(post_folder_data, post_folder_data, n_dti=interval, n_min=25)
        print(f'Linear interpolation post-processing applied, saved to {post_folder_data}.')

        if effective_post_mode in {'post_gbi', 'both'}:
            post_folder_gbi = os.path.join(args.result_folder, args.exp_name + '_post_gbi', 'data')
            os.makedirs(post_folder_gbi, exist_ok=True)
            for file_name in os.listdir(post_folder_data):
                in_path = os.path.join(post_folder_data, file_name)
                out_path = os.path.join(post_folder_gbi, file_name)
                GBInterpolation(path_in=in_path, path_out=out_path, interval=interval)
            print(f'Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.')

    if args.make_submission:
        submit_post = default_post if args.post_mode in {'auto', 'both'} else effective_post_mode
        if submit_post == 'none':
            submit_name = args.exp_name
        else:
            submit_name = args.exp_name + '_' + submit_post
        input_dir = os.path.join(args.result_folder, submit_name, 'data')
        output_zip = args.submission_zip or os.path.join('results', f'{submit_name}_submission.zip')
        _flat_zip(input_dir, output_zip)


if __name__ == '__main__':
    main()
