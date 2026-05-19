# Model Weights

Place detector and ReID weights in this directory, or pass absolute paths with
`--detector_path` and `--reid_path`.

Expected local filenames:

| Dataset | Validation detector | Test detector | Validation ReID | Test ReID |
| --- | --- | --- | --- | --- |
| MOT17 | `bytetrack_ablation.pth.tar` | `bytetrack_x_mot17.pth.tar` | `osnet_ain_ms_d_c.pth.tar` | `mot17_sbs_S50.pth` |
| MOT20 | `bytetrack_x_mot17.pth.tar` | `bytetrack_x_mot20.tar` | `osnet_ain_ms_d_c.pth.tar` | `mot20_sbs_S50.pth` |
| DanceTrack | `bytetrack_dance_model.pth.tar` | `dance.pth.tar` | `dance_sbs_S50.pth` | `dance_sbs_S50.pth` |
| SportsMOT | `SportsMOT_yolox_x.tar` | `SportsMOT_yolox_x.tar` or `SportsMOT_yolox_x_mix.tar` | `sports_sbs_S50.pth` | `sports_sbs_S50.pth` |

For SportsMOT, `SportsMOT_yolox_x_mix.tar` corresponds to the stronger detector
setting used for the extra-data/stronger-detector comparison.
