#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
SPORTSMOT_DATA_DIR="${SPORTSMOT_DATA_DIR:-data/sportsmot_publish/dataset}"

check_zip() {
  local zip_path="$1"
  local expected="$2"
  python - "$zip_path" "$expected" <<'PY'
import sys, zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not zip_path.is_file():
    raise SystemExit(f"Missing zip: {zip_path}")
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
txt = [name for name in names if name.endswith(".txt")]
nested = [name for name in names if "/" in name.strip("/")]
if nested:
    raise SystemExit(f"{zip_path.name} is not flat: {nested[:3]}")
if len(txt) != expected:
    raise SystemExit(f"{zip_path.name}: expected {expected} txt files, found {len(txt)}")
print(f"Checked {zip_path}: {len(txt)} flat txt files.")
PY
}

python main.py \
  --dataset mot17 \
  --data_dir "$DATA_DIR" \
  --test_dataset \
  --all_mot17_detectors \
  --exp_name MemoTrack_MOT17_test \
  --make_submission
check_zip results/MemoTrack_MOT17_test_post_gbi_submission.zip 21

python main.py \
  --dataset mot20 \
  --data_dir "$DATA_DIR" \
  --test_dataset \
  --exp_name MemoTrack_MOT20_test \
  --make_submission
check_zip results/MemoTrack_MOT20_test_post_gbi_submission.zip 4

python main.py \
  --dataset dance \
  --data_dir "$DATA_DIR" \
  --test_dataset \
  --detector_path external/weights/dance.pth.tar \
  --reid_path external/weights/dance_sbs_S50.pth \
  --exp_name MemoTrack_DANCE_test \
  --make_submission
check_zip results/MemoTrack_DANCE_test_post_submission.zip 35

python main.py \
  --dataset sportsmot \
  --data_dir "$SPORTSMOT_DATA_DIR" \
  --test_dataset \
  --detector_path external/weights/SportsMOT_yolox_x.tar \
  --reid_path external/weights/sports_sbs_S50.pth \
  --exp_name MemoTrack_SPORTSMOT_test \
  --make_submission
check_zip results/MemoTrack_SPORTSMOT_test_post_submission.zip 150

python main.py \
  --dataset sportsmot \
  --data_dir "$SPORTSMOT_DATA_DIR" \
  --test_dataset \
  --detector_path external/weights/SportsMOT_yolox_x_mix.tar \
  --reid_path external/weights/sports_sbs_S50.pth \
  --exp_name MemoTrack_SPORTSMOT_test_mix \
  --make_submission
check_zip results/MemoTrack_SPORTSMOT_test_mix_post_submission.zip 150
