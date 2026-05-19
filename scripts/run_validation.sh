#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
GT_DIR="${GT_DIR:-results/gt}"
SPORTSMOT_DATA_DIR="${SPORTSMOT_DATA_DIR:-data/sportsmot_publish/dataset}"

run_mot_eval() {
  local benchmark="$1"
  local tracker="$2"
  python external/TrackEval/scripts/run_mot_challenge.py \
    --SPLIT_TO_EVAL val \
    --GT_FOLDER "$GT_DIR" \
    --TRACKERS_FOLDER results/trackers \
    --BENCHMARK "$benchmark" \
    --TRACKERS_TO_EVAL "$tracker" \
    --USE_PARALLEL False \
    --METRICS HOTA CLEAR Identity \
    --PLOT_CURVES False \
    --PRINT_CONFIG False \
    --PRINT_ONLY_COMBINED True
}

python main.py --dataset mot17 --data_dir "$DATA_DIR" --exp_name MemoTrack_MOT17_val --post_mode post_gbi
run_mot_eval MOT17 MemoTrack_MOT17_val_post_gbi

python main.py --dataset mot20 --data_dir "$DATA_DIR" --exp_name MemoTrack_MOT20_val --post_mode post_gbi
run_mot_eval MOT20 MemoTrack_MOT20_val_post_gbi

python main.py \
  --dataset dance \
  --data_dir "$DATA_DIR" \
  --detector_path external/weights/dance.pth.tar \
  --reid_path external/weights/dance_sbs_S50.pth \
  --exp_name MemoTrack_DANCE_val \
  --post_mode post_gbi
run_mot_eval DANCE MemoTrack_DANCE_val_post_gbi

python main.py \
  --dataset sportsmot \
  --data_dir "$SPORTSMOT_DATA_DIR" \
  --detector_path external/weights/SportsMOT_yolox_x.tar \
  --reid_path external/weights/sports_sbs_S50.pth \
  --exp_name MemoTrack_SPORTSMOT_val \
  --post_mode post
python eval_sportsmot.py \
  --data_root "$SPORTSMOT_DATA_DIR" \
  --trackers_folder results/trackers/SPORTSMOT-val \
  --tracker_name MemoTrack_SPORTSMOT_val_post
