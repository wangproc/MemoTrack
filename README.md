# MemoTrack

MemoTrack is an online multi-object tracker that uses the PHD posterior as a
queryable existence-evidence field. Instead of replacing the Kalman filter or
extracting identities from a full RFS tracker, MemoTrack keeps the standard
tracking-by-detection pipeline and injects posterior evidence into association.

The tracker contains two complementary modules:

- **Posterior Track Calibration (PTC):** queries posterior support at predicted
  track locations to refine track-side reliability during high-score matching.
- **Track-Conditioned Evidence Recovery (TCER):** recovers reliable low-score
  observations when they are supported by posterior existence evidence,
  short-term lost tracks, and local structural safety.

## Repository Layout

```text
MemoTrack
|-- main.py                    # unified entry for validation and test submission
|-- memotracker/               # MemoTrack core, PHD posterior, PTC, and TCER
|-- external/                  # detector, ReID, and TrackEval dependencies
|-- data/tools/                # dataset conversion helpers
|-- tools/                     # qualitative visualization and PHD heatmap tools
|-- results/                   # generated tracking outputs and submissions
|-- environment.yml            # conda environment
```

Datasets, model weights, caches, and generated results are intentionally ignored
by git.

## Installation

The code was tested on Ubuntu 22.04 with CUDA, PyTorch, YOLOX, FastReID, and
TrackEval.

```bash
conda env create -f environment.yml
conda activate boostTrack
```

Prepare datasets under `data/` following the standard benchmark layout:

```text
data/
|-- MOT17/
|   |-- train/
|   `-- test/
|-- MOT20/
|   |-- train/
|   `-- test/
|-- dancetrack/
|   |-- train/
|   |-- val/
|   `-- test/
`-- sportsmot_publish/
    `-- dataset/
```

Place model weights in `external/weights/`. See
`external/weights/README.md` for the expected filenames.

## Validation

To run the full validation pipeline, set dataset roots if they are outside this
repository and launch:

```bash
DATA_DIR=/path/to/data \
GT_DIR=/path/to/results/gt \
SPORTSMOT_DATA_DIR=/path/to/sportsmot_publish/dataset \
bash scripts/run_validation.sh
```

The individual commands are listed below for easier debugging.

Run MemoTrack on MOT17 validation:

```bash
python main.py --dataset mot17 --exp_name MemoTrack_MOT17_val --post_mode post_gbi
python external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --GT_FOLDER results/gt \
  --TRACKERS_FOLDER results/trackers \
  --BENCHMARK MOT17 \
  --TRACKERS_TO_EVAL MemoTrack_MOT17_val_post_gbi \
  --USE_PARALLEL False \
  --METRICS HOTA CLEAR Identity \
  --PLOT_CURVES False \
  --PRINT_ONLY_COMBINED True
```

Run MOT20 validation:

```bash
python main.py --dataset mot20 --exp_name MemoTrack_MOT20_val --post_mode post_gbi
python external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --GT_FOLDER results/gt \
  --TRACKERS_FOLDER results/trackers \
  --BENCHMARK MOT20 \
  --TRACKERS_TO_EVAL MemoTrack_MOT20_val_post_gbi \
  --USE_PARALLEL False \
  --METRICS HOTA CLEAR Identity \
  --PLOT_CURVES False \
  --PRINT_ONLY_COMBINED True
```

DanceTrack and SportsMOT can be run through the same entry:

```bash
python main.py --dataset dance --exp_name MemoTrack_DANCE_val --post_mode post
python main.py --dataset sportsmot \
  --data_dir data/sportsmot_publish/dataset \
  --exp_name MemoTrack_SPORTSMOT_val \
  --post_mode post
```

## Test Submission

To generate all test submission files:

```bash
DATA_DIR=/path/to/data \
SPORTSMOT_DATA_DIR=/path/to/sportsmot_publish/dataset \
bash scripts/run_test_submissions.sh
```

MOT17 uses all three public detector splits on the test set:

```bash
python main.py \
  --dataset mot17 \
  --test_dataset \
  --all_mot17_detectors \
  --exp_name MemoTrack_MOT17_test \
  --make_submission
```

MOT20:

```bash
python main.py \
  --dataset mot20 \
  --test_dataset \
  --exp_name MemoTrack_MOT20_test \
  --make_submission
```

DanceTrack:

```bash
python main.py \
  --dataset dance \
  --test_dataset \
  --detector_path external/weights/dance.pth.tar \
  --reid_path external/weights/dance_sbs_S50.pth \
  --exp_name MemoTrack_DANCE_test \
  --make_submission
```

SportsMOT standard detector:

```bash
python main.py \
  --dataset sportsmot \
  --test_dataset \
  --data_dir data/sportsmot_publish/dataset \
  --detector_path external/weights/SportsMOT_yolox_x.tar \
  --reid_path external/weights/sports_sbs_S50.pth \
  --exp_name MemoTrack_SPORTSMOT_test \
  --make_submission
```

SportsMOT stronger detector setting:

```bash
python main.py \
  --dataset sportsmot \
  --test_dataset \
  --data_dir data/sportsmot_publish/dataset \
  --detector_path external/weights/SportsMOT_yolox_x_mix.tar \
  --reid_path external/weights/sports_sbs_S50.pth \
  --exp_name MemoTrack_SPORTSMOT_test_mix \
  --make_submission
```

Each `--make_submission` run creates a flat zip file under `results/`.

## Visualization

Render tracking boxes:

```bash
python tools/visualize_tracking_result.py \
  --dataset mot17 \
  --result_dir results/trackers/MOT17-val/MemoTrack_MOT17_val_post_gbi/data \
  --output_dir results/img_result/qualitative/mot17
```

Render PHD intensity overlays:

```bash
python tools/visualize_phd_intensity.py \
  --dataset mot17 \
  --exp_name MemoTrack_heatmap \
  --output_dir results/img_result/phd_intensity/mot17
```

## Acknowledgements

This repository uses public components from the MOT ecosystem, including YOLOX,
FastReID, TrackEval, ByteTrack-style detectors, and related online tracking
utilities. We thank the authors of these projects for making their code and
models available to the community.

## Citation

If MemoTrack is useful for your research, please cite:

```bibtex
@article{memotrack,
  title={MemoTrack: Online Multi-Object Tracking by Existence Query},
  author={},
  journal={},
  year={}
}
```
