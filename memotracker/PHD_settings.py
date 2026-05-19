from typing import Union, Dict


class RFSSettings:
    values: Dict[str, Union[float, bool, int, str]] = {
        # ── PHD filter core parameters ──────────────────────────────────────
        'x_dim': 8,                 # state dim [cx, cy, h, a, vcx, vcy, vh, va]
        'z_dim': 4,                 # observation dim [cx, cy, h, a]
        'P_S': 0.99,               # survival probability
        'P_D': 0.60,               # detection probability
        'lambda_c': 15,            # clutter intensity
        'L_max': 200,              # max Gaussian components
        'elim_threshold': 1e-6,    # pruning threshold
        'merge_threshold': 7,      # GM merging threshold (7=conservative, keeps nearby people separate)
        'extract_threshold': 0.5,  # state extraction weight threshold (GM-PHD extract)
        'gate_flag': True,         # enable gating
        'P_G': 0.999,              # gating probability

        # ── birth model ─────────────────────────────────────────────────────
        'birth_weight_coef': 0.08, # birth weight = det_score × coef
        'birth_P_scale': 100.0,    # initial covariance scale for birth
        'use_novelty_aware_birth': False,
        'birth_support_suppress': 0.28,
        'birth_novelty_floor': 0.30,
        'birth_novelty_topk': 2,
        'birth_novelty_maha_gate': 9.0,
        'birth_novelty_dominant_gate': 0.75,

        # ── PHD weight tracking ──────────────────────────────────────────────
        # phd_smooth_alpha: EMA coefficient for update_phd_weight.
        # alpha=0.5: new_w = 0.5×new + 0.5×old → halves each miss frame.
        'phd_smooth_alpha': 0.5,
        'use_soft_phd_weight': False,
        'phd_weight_topk': 3,
        'phd_weight_maha_gate': 16.0,
        'phd_weight_ambiguity_floor': 0.80,
        'phd_weight_cap': 1.10,
        'use_supportive_low_score_update': False,
        'phd_update_low_score_thresh': 0.45,
        'phd_update_track_iou_gate': 0.18,
        'phd_update_pd_low_floor': 0.35,
        'phd_update_pd_low_power': 1.5,
        'use_track_pseudo_update': True,
        'phd_pseudo_low_score_thresh': 0.35,
        'phd_pseudo_high_iou_suppress': 0.15,
        'phd_pseudo_medium_iou_gate': 0.12,
        'phd_pseudo_blend_alpha': 0.35,
        'phd_pseudo_track_conf_min': 0.70,
        'phd_pseudo_phd_min': 0.45,
        'phd_pseudo_supported_low_score_thresh': 0.30,
        'phd_pseudo_supported_iou_gate': 0.08,
        'phd_pseudo_supported_track_conf_min': 0.60,
        'phd_pseudo_supported_phd_min': 0.35,
        'phd_pseudo_supported_max_age': 2,
        'phd_pseudo_supported_min_hits': 2,
        'phd_pseudo_supported_blend_alpha': 0.42,
        'phd_pseudo_supported_high_iou_suppress': 0.38,
        'phd_pseudo_supported_high_iou_margin': 0.15,
        'phd_pseudo_supported_bypass_score': 0.42,
        'phd_pseudo_supported_bypass_iou': 0.12,
        'phd_pseudo_supported_strict_high_iou_suppress': 0.55,
        'phd_pseudo_supported_strict_margin': 0.25,
        'phd_pseudo_score_base': 0.44,
        'phd_pseudo_score_medium_bonus': 0.06,
        'phd_pseudo_score_track_gain': 0.05,
        'phd_pseudo_score_phd_gain': 0.04,
        'phd_pseudo_score_max': 0.58,
        'phd_pseudo_max_age': 1,
        'phd_pseudo_max_per_frame': 8,
        'phd_pseudo_nms_iou': 0.50,

        # ── PHD confidence boost (Innovation 2, v7: lost tracks only) ────────
        # Low-score detections near LOST tracks are evaluated against PHD intensity.
        # If PHD intensity >= 1e-4, detection is boosted to det_thresh+0.1.
        # Restricts boost to detections near lost tracks (not active tracks),
        # so boosted detections flow to Stage 2 without polluting Stage 1.
        'use_phd_boost': True,
        'boost_iou_gate': 0.3,     # min IoU with lost track to be eligible for boost

        # ── Stage 2 (PHD-gated lost track recovery, Innovation 3) ────────────
        # Only lost tracks with phd_weight >= phd_stage2_gate enter Stage 2.
        # PHD confirmation prevents spurious ghost track associations.
        'phd_stage2_gate': 0.005,   # min phd_weight for Stage 2 participation
        'stage2_threshold': 0.28,  # IoU threshold for Stage 2 matching
        'use_stage2': True,

        # ── Stage 2b (disabled) ──────────────────────────────────────────────
        'use_stage2b': False,
        'det_thresh_low': 0.45,
        'stage2b_threshold': 0.30,

        # ── association parameters ───────────────────────────────────────────
        'lambda_iou': 0.5,
        'lambda_mhd': 0.25,
        'lambda_shape': 0.25,
        'stage1_threshold': 0.3,
        'traj_l_scan_stage1': 4,
        'stage1_traj_stability_weight': 0.20,
        'stage1_competition_floor': 0.40,
        'stage1_competition_age_decay': 0.20,
        'stage1_competition_hit_bonus': 0.05,

        # ── track management ─────────────────────────────────────────────────
        'max_age': 30,
        'min_hits': 3,
        'det_thresh': 0.6,         # detection confidence threshold (mot17)

        # ── ReID and ECC (reuse original) ────────────────────────────────────
        'use_embedding': True,
        'use_ecc': True,
    }

    dataset_specific_settings: Dict[str, Dict[str, Union[float, bool, int]]] = {
        "mot17": {"det_thresh": 0.6},
        "mot20": {"det_thresh": 0.4, "lambda_c": 20, "L_max": 300},
    }

    @staticmethod
    def __class_getitem__(key: str):
        from default_settings import GeneralSettings
        dataset = GeneralSettings.values.get('dataset', 'mot17')
        try:
            return RFSSettings.dataset_specific_settings[dataset][key]
        except (KeyError, TypeError):
            return RFSSettings.values[key]
