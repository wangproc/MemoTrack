from __future__ import print_function

from copy import deepcopy
from typing import List, Optional

import numpy as np

from default_settings import GeneralSettings, BoostTrackSettings
from .PHD_assoc import iou_batch, rfs_associate_stage1, rfs_associate_stage2, phd_confidence_boost
from .PHD_dataset_config import PHDDatasetConfig
from .PHD_filter import PHDFilter, convert_bbox_to_z as phd_convert_bbox_to_z
from .PHD_settings import RFSSettings
from .embedding import EmbeddingComputer
from .ecc import ECC
from .kalmanbox import KalmanBoxTracker


BASE_RFS_VALUES = deepcopy(RFSSettings.values)


FINAL_CFG = {
    'high_det_thresh': 0.6,
    'low_det_thresh': 0.3,
    'init_thresh': 0.8,
    'first_match_iou': 0.2,
    'second_match_iou': 0.6,
    'unconfirmed_match_iou': 0.3,
    'use_ecc': True,
    'use_assoc_emb': True,
    'use_assoc_mhd': False,
    'use_assoc_shape': False,
    'use_phd_core': True,
    'use_A': True,
    'use_B': False,
    'use_C': False,
    'use_E': False,
    'use_C_prime': True,
    'use_active_rescue': False,
    'use_conf_boost': False,
    'use_full_rfs_stage1': True,
    'use_A_pairwise': False,
    'use_B_prime_group': True,
    'use_B_prime_plus': False,
    'use_B_prime': False,
    'use_bprime_group_phd_rescue': False,
    'use_bprime_group_phd_targeted': False,
    'use_bprime_group_anchor': False,
    'enable_b_diag': False,
    'enable_cprime_diag': False,
    'cprime_low_det_thresh': 0.3,
    'cprime_lost_max_age': 3,
    'cprime_track_phd_gate': 0.03,
    'cprime_boost_iou_gate': 0.15,
    'cprime_stage2_threshold': 0.22,
    'cprime_stage2_phd_soft_bias': 0.10,
    'cprime_stage2_emb_weight': 1.50,
    'bprime_group_active_iou_gate': 0.45,
    'bprime_group_second_lost_iou_gate': 0.28,
    'bprime_group_min_best_iou': 0.2,
    'bprime_group_best_weight': 0.55,
    'bprime_group_shared_weight': 1.0,
    'bprime_group_dominance_weight': 1.2,
    'bprime_group_min_score': 0.5,
    'bprime_group_phd_best_iou_gate': 0.38,
    'bprime_group_phd_track_gate': 0.37,
    'bprime_group_phd_intensity_gate': 0.53,
    'bprime_group_phd_active_max': 0.18,
    'bprime_group_phd_margin_gate': 0.18,
    'bprime_group_phd_dominant_share_gate': 0.6,
    'bprime_group_phd_stage2_threshold': 0.28,
    'bprime_group_phd_stage2_phd_soft_bias': 0.10,
    'bprime_group_phd_stage2_emb_weight': 1.50,
    'bprime_group_anchor_min_best_iou': 0.45,
    'bprime_group_anchor_max_active_iou': 0.18,
    'bprime_group_anchor_min_margin': 0.22,
    'bprime_group_anchor_stage2_threshold': 0.30,
    'stage1_competition_floor': 0.4,
    'stage1_competition_age_decay': 0.2,
    'stage1_competition_hit_bonus': 0.05,
}


class MemoTracklet:
    def __init__(self, bbox: np.ndarray, emb: Optional[np.ndarray] = None, phd_weight: float = 0.5):
        self.kf_tracker = KalmanBoxTracker(bbox, emb=emb)
        self.bbox_to_z_func = self.kf_tracker.bbox_to_z_func
        self.kf = self.kf_tracker.kf
        self.phd_weight = float(phd_weight)
        self.phd_component_id = -1
        self.phd_support_count = 0
        self.phd_dominant_share = 0.0
        self.traj_history = []
        self._push_history(bbox[:4])

    @property
    def id(self):
        return self.kf_tracker.id

    @property
    def time_since_update(self):
        return self.kf_tracker.time_since_update

    @property
    def hit_streak(self):
        return self.kf_tracker.hit_streak

    @property
    def age(self):
        return self.kf_tracker.age

    def predict(self):
        return self.kf_tracker.predict()

    def get_state(self):
        return self.kf_tracker.get_state()

    def camera_update(self, transform: np.ndarray):
        self.kf_tracker.camera_update(transform)

    def update(self, det_bbox: np.ndarray, score: float):
        self.kf_tracker.update(det_bbox, score)
        self._push_history(det_bbox[:4])

    def get_confidence(self, coef: float = 0.9) -> float:
        return self.kf_tracker.get_confidence(coef)

    def update_emb(self, emb, alpha: float = 0.9):
        if emb is None:
            return
        if self.kf_tracker.get_emb() is None:
            self.kf_tracker.emb = emb
        else:
            self.kf_tracker.update_emb(emb, alpha)

    def get_emb(self):
        return self.kf_tracker.get_emb()

    def _push_history(self, bbox: np.ndarray):
        self.traj_history.append(np.asarray(bbox[:4], dtype=float).copy())
        max_len = max(2, int(RFSSettings.values.get('traj_l_scan_stage1', 4)))
        if len(self.traj_history) > max_len:
            self.traj_history = self.traj_history[-max_len:]

    def get_traj_history(self) -> np.ndarray:
        if len(self.traj_history) == 0:
            return np.empty((0, 4), dtype=float)
        return np.asarray(self.traj_history, dtype=float)

    def update_phd_weight(self, phd_filter, smooth_alpha: float = 0.4):
        z = phd_convert_bbox_to_z(self.get_state()[0])
        nearest_w = 0.0
        nearest_comp = -1
        support_count = 0
        dominant_share = 0.0

        if RFSSettings.values.get('use_soft_phd_weight', True):
            support = phd_filter.query_posterior_support(
                z,
                topk=int(RFSSettings.values.get('phd_weight_topk', 3)),
                maha_gate=float(RFSSettings.values.get('phd_weight_maha_gate', 16.0)),
                ambiguity_floor=float(RFSSettings.values.get('phd_weight_ambiguity_floor', 0.80)),
                weight_cap=float(RFSSettings.values.get('phd_weight_cap', 1.10)),
            )
            nearest_w = float(support['soft_weight'])
            nearest_comp = int(support['component_id'])
            support_count = int(support['support_count'])
            dominant_share = float(support['dominant_share'])

        if nearest_comp < 0 and phd_filter.m_update.shape[1] > 0:
            z_col = z.reshape(4, 1)
            dists = np.sum((phd_filter.m_update[:2, :] - z_col[:2, :]) ** 2, axis=0)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] < 2000:
                nearest_w = float(phd_filter.w_update[best_idx])
                nearest_comp = int(best_idx)
                support_count = 1
                dominant_share = 1.0

        self.phd_weight = smooth_alpha * nearest_w + (1 - smooth_alpha) * getattr(self, 'phd_weight', 0.5)
        self.phd_component_id = nearest_comp
        self.phd_support_count = support_count
        self.phd_dominant_share = dominant_share


class MemoTracker:
    def __init__(self, video_name: Optional[str] = None, cfg_override: Optional[dict] = None):
        cfg = dict(FINAL_CFG)
        cfg.update(PHDDatasetConfig.tracker_cfg(GeneralSettings.values.get('dataset', 'mot17')))
        if cfg_override:
            cfg.update(cfg_override)
        # Final clean release line: rescue-only variants are excluded from MemoTrack.
        cfg['use_bprime_group_phd_rescue'] = False
        self.frame_count = 0
        self.trackers: List[MemoTracklet] = []
        self.video_name = video_name
        self.cfg = cfg

        self.max_age = GeneralSettings.max_age(video_name)
        self.min_hits = GeneralSettings['min_hits']
        self.det_thresh = float(cfg.get('det_thresh', GeneralSettings['det_thresh']))
        self.boost_iou_threshold = GeneralSettings['iou_threshold']

        self.use_ecc = bool(cfg.get('use_ecc', True))
        self.use_assoc_emb = bool(cfg.get('use_assoc_emb', True))
        self.use_assoc_mhd = bool(cfg.get('use_assoc_mhd', False))
        self.use_assoc_shape = bool(cfg.get('use_assoc_shape', False))
        self.use_A = bool(cfg.get('use_A', True))
        self.use_C_prime = bool(cfg.get('use_C_prime', True))
        self.use_any_bprime = bool(cfg.get('use_B_prime_group', True))
        self.use_bprime_group_phd_rescue = bool(cfg.get('use_bprime_group_phd_rescue', False))
        self.use_bprime_group_anchor = bool(cfg.get('use_bprime_group_anchor', False))

        self.lambda_iou = float(cfg.get('lambda_iou', BoostTrackSettings['lambda_iou']))
        self.lambda_mhd = float(cfg.get('lambda_mhd', BoostTrackSettings['lambda_mhd']))
        self.lambda_shape = float(cfg.get('lambda_shape', BoostTrackSettings['lambda_shape']))

        self.embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True)
        self.ecc = ECC(scale=350, video_name=video_name, use_cache=True) if self.use_ecc else None

        rfs_values = deepcopy(BASE_RFS_VALUES)
        rfs_values.update(RFSSettings.dataset_specific_settings.get(GeneralSettings.values.get('dataset', 'mot17'), {}))
        for key in (
            'P_S', 'P_D', 'lambda_c', 'L_max', 'elim_threshold', 'merge_threshold', 'extract_threshold',
            'gate_flag', 'P_G', 'birth_weight_coef', 'birth_P_scale', 'use_novelty_aware_birth',
            'birth_support_suppress', 'birth_novelty_floor', 'birth_novelty_topk', 'birth_novelty_maha_gate',
            'birth_novelty_dominant_gate', 'phd_smooth_alpha', 'use_soft_phd_weight', 'phd_weight_topk',
            'phd_weight_maha_gate', 'phd_weight_ambiguity_floor', 'phd_weight_cap',
            'use_supportive_low_score_update', 'phd_update_low_score_thresh', 'phd_update_track_iou_gate',
            'phd_update_pd_low_floor', 'phd_update_pd_low_power', 'use_track_pseudo_update',
        ):
            if key in cfg:
                rfs_values[key] = cfg[key]
        RFSSettings.values.clear()
        RFSSettings.values.update(rfs_values)

        self.phd = PHDFilter(rfs_values, fov_size=(1080, 1920))
        self.phd_smooth_alpha = float(RFSSettings.values.get('phd_smooth_alpha', 0.5))
        self.cprime_low_det_thresh = float(cfg['cprime_low_det_thresh'])
        self.cprime_lost_max_age = int(cfg['cprime_lost_max_age'])
        self.cprime_track_phd_gate = float(cfg['cprime_track_phd_gate'])
        self.cprime_boost_iou_gate = float(cfg['cprime_boost_iou_gate'])
        self.cprime_stage2_threshold = float(cfg['cprime_stage2_threshold'])
        self.cprime_stage2_phd_soft_bias = float(cfg['cprime_stage2_phd_soft_bias'])
        self.cprime_stage2_emb_weight = float(cfg['cprime_stage2_emb_weight'])
        self.bprime_group_active_iou_gate = float(cfg['bprime_group_active_iou_gate'])
        self.bprime_group_second_lost_iou_gate = float(cfg['bprime_group_second_lost_iou_gate'])
        self.bprime_group_min_best_iou = float(cfg['bprime_group_min_best_iou'])
        self.bprime_group_best_weight = float(cfg['bprime_group_best_weight'])
        self.bprime_group_shared_weight = float(cfg['bprime_group_shared_weight'])
        self.bprime_group_dominance_weight = float(cfg['bprime_group_dominance_weight'])
        self.bprime_group_min_score = float(cfg['bprime_group_min_score'])
        self.bprime_group_phd_best_iou_gate = float(cfg['bprime_group_phd_best_iou_gate'])
        self.bprime_group_phd_track_gate = float(cfg['bprime_group_phd_track_gate'])
        self.bprime_group_phd_intensity_gate = float(cfg['bprime_group_phd_intensity_gate'])
        self.bprime_group_phd_active_max = float(cfg['bprime_group_phd_active_max'])
        self.bprime_group_phd_margin_gate = float(cfg['bprime_group_phd_margin_gate'])
        self.bprime_group_phd_dominant_share_gate = float(cfg['bprime_group_phd_dominant_share_gate'])
        self.bprime_group_phd_stage2_threshold = float(cfg['bprime_group_phd_stage2_threshold'])
        self.bprime_group_phd_stage2_phd_soft_bias = float(cfg['bprime_group_phd_stage2_phd_soft_bias'])
        self.bprime_group_phd_stage2_emb_weight = float(cfg['bprime_group_phd_stage2_emb_weight'])
        self.stage1_competition_floor = float(cfg['stage1_competition_floor'])
        self.stage1_competition_age_decay = float(cfg['stage1_competition_age_decay'])
        self.stage1_competition_hit_bonus = float(cfg['stage1_competition_hit_bonus'])
        self.traj_l_scan_stage1 = int(RFSSettings.values.get('traj_l_scan_stage1', 4))

        self._bprime_diag = {
            'frames': 0,
            'promoted_candidates': 0,
            'kept_promoted_candidates': 0,
            'filtered_promoted_candidates': 0,
            'phd_rescue_candidates': 0,
            'phd_rescue_matched': 0,
        }

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    @staticmethod
    def _stack_measurements(dets: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            return np.empty((4, 0), dtype=float)
        cols = [phd_convert_bbox_to_z(det[:4]).reshape(4, 1) for det in dets]
        return np.concatenate(cols, axis=1)

    def _compute_bprime_group_stats(self, active_ious: np.ndarray, lost_ious_all: np.ndarray):
        if lost_ious_all.size == 0:
            num_rows = active_ious.shape[0] if active_ious.ndim > 0 else 0
            zeros = np.zeros(num_rows, dtype=float)
            return zeros, zeros, zeros, zeros, zeros

        num_rows = lost_ious_all.shape[0]
        if active_ious.size > 0:
            active_max = np.max(active_ious, axis=1)
        else:
            active_max = np.zeros(num_rows, dtype=float)
        best_lost = np.max(lost_ious_all, axis=1)
        if lost_ious_all.shape[1] >= 2:
            top_sorted = np.sort(lost_ious_all, axis=1)
            second_lost = top_sorted[:, -2]
        else:
            second_lost = np.zeros(num_rows, dtype=float)

        dominance = np.maximum(best_lost - active_max, 0.0)
        structure_score = (
            self.bprime_group_best_weight * best_lost
            + self.bprime_group_shared_weight * second_lost
            + self.bprime_group_dominance_weight * dominance
        )
        return active_max, best_lost, second_lost, dominance, structure_score

    def _select_bprime_group_promotions(self, active_ious: np.ndarray, lost_ious_all: np.ndarray) -> np.ndarray:
        if lost_ious_all.size == 0:
            return np.zeros((active_ious.shape[0] if active_ious.ndim > 0 else 0,), dtype=bool)

        active_max, best_lost, second_lost, _, structure_score = self._compute_bprime_group_stats(active_ious, lost_ious_all)
        return (
            (best_lost >= self.bprime_group_min_best_iou)
            & ((active_max >= self.bprime_group_active_iou_gate) | (second_lost >= self.bprime_group_second_lost_iou_gate))
            & (structure_score >= self.bprime_group_min_score)
        )

    def _select_bprime_group_phd_rescue(
        self,
        active_ious: np.ndarray,
        lost_ious_all: np.ndarray,
        promoted_margins: np.ndarray,
        promoted_intensity_norm: np.ndarray,
        lost_track_phd: np.ndarray,
        lost_track_dom_share: np.ndarray,
        hard_keep_mask: np.ndarray,
    ) -> np.ndarray:
        if (not self.use_bprime_group_phd_rescue) or lost_ious_all.size == 0:
            return np.zeros((active_ious.shape[0] if active_ious.ndim == 2 else 0,), dtype=bool)

        active_max, best_lost, second_lost, _, _ = self._compute_bprime_group_stats(active_ious, lost_ious_all)
        best_idx = np.argmax(lost_ious_all, axis=1)
        best_track_phd = lost_track_phd[best_idx]
        best_track_dom = lost_track_dom_share[best_idx]
        open_mask = ~hard_keep_mask
        return (
            open_mask
            & (best_lost >= self.bprime_group_phd_best_iou_gate)
            & (best_track_phd >= self.bprime_group_phd_track_gate)
            & (best_track_dom >= self.bprime_group_phd_dominant_share_gate)
            & (promoted_intensity_norm >= self.bprime_group_phd_intensity_gate)
            & (active_max <= self.bprime_group_phd_active_max)
            & (promoted_margins >= self.bprime_group_phd_margin_gate)
            & (second_lost < self.bprime_group_second_lost_iou_gate)
        )

    @staticmethod
    def _normalize_phd_intensity(intensity: np.ndarray, base: float = 1e-4, high: float = 2e-3) -> np.ndarray:
        vals = np.asarray(intensity, dtype=float)
        ratio = np.maximum(vals / max(base, 1e-12), 1.0)
        scale = max(np.log(high / max(base, 1e-12)), 1e-6)
        return np.clip(np.log(ratio) / scale, 0.0, 1.0)

    def update(self, dets, img_tensor, img_numpy, tag):
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)
            self.phd.camera_update(transform)

        tracker_boxes = np.zeros((len(self.trackers), 5), dtype=float)
        tracker_confs = np.zeros((len(self.trackers), 1), dtype=float)
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            conf = trk.get_confidence()
            tracker_confs[i, 0] = conf
            tracker_boxes[i] = [pos[0], pos[1], pos[2], pos[3], conf]

        self.phd.predict()

        high_mask = dets[:, 4] >= self.det_thresh
        high_dets = dets[high_mask]
        cprime_low_mask = (dets[:, 4] >= self.cprime_low_det_thresh) & (dets[:, 4] < self.det_thresh)
        cprime_low_ids = np.flatnonzero(cprime_low_mask)
        cprime_low_dets = dets[cprime_low_ids] if len(dets) > 0 else np.empty((0, 5))

        meas_z = self._stack_measurements(high_dets)
        meas_scores = None if len(high_dets) == 0 else high_dets[:, 4]
        self.phd.update(meas_z, measurement_scores=meas_scores)
        for trk in self.trackers:
            trk.update_phd_weight(self.phd, smooth_alpha=self.phd_smooth_alpha)

        if len(high_dets) > 0:
            high_embs = self.embedder.compute_embedding(img_numpy, high_dets[:, :4], tag)
        else:
            high_embs = np.empty((0, 1), dtype=float)

        trk_embs = [trk.get_emb() for trk in self.trackers]
        trk_embs = np.array(trk_embs) if len(trk_embs) > 0 else np.empty((0,))
        emb_cost = None
        if trk_embs.size > 0 and high_embs is not None and high_embs.size > 0:
            emb_cost = high_embs.reshape(high_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T

        phd_weights_arr = np.array([trk.phd_weight for trk in self.trackers], dtype=float)
        stage1_phd_weights = phd_weights_arr if self.use_A else None
        track_histories = [trk.get_traj_history() for trk in self.trackers]
        track_ages = np.array([trk.time_since_update for trk in self.trackers], dtype=int)
        track_hit_streaks = np.array([trk.hit_streak for trk in self.trackers], dtype=int)
        base_track_conf = tracker_confs.ravel().astype(float)

        if len(high_dets) > 0:
            matches, unmatched_high, unmatched_trackers, _ = rfs_associate_stage1(
                high_dets,
                tracker_boxes,
                stage1_phd_weights,
                mahalanobis_distance=None,
                track_confidence=base_track_conf,
                detection_confidence=high_dets[:, 4],
                emb_cost=emb_cost,
                iou_threshold=self.boost_iou_threshold,
                lambda_iou=self.lambda_iou,
                lambda_mhd=0.0,
                lambda_shape=0.0,
                track_histories=track_histories,
                track_ages=track_ages,
                track_hit_streaks=track_hit_streaks,
                phd_component_ids=None,
                traj_l_scan=self.traj_l_scan_stage1,
                traj_stability_weight=0.0,
                competition_floor=self.stage1_competition_floor,
                competition_age_decay=self.stage1_competition_age_decay,
                competition_hit_bonus=self.stage1_competition_hit_bonus,
                track_confidence_precomputed=False,
            )
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_high = np.empty((0,), dtype=int)
            unmatched_trackers = np.arange(len(self.trackers), dtype=int)

        matched_tracker_ids = set()
        for det_idx, trk_idx in matches:
            det_idx = int(det_idx)
            trk_idx = int(trk_idx)
            self.trackers[trk_idx].update(high_dets[det_idx, :], high_dets[det_idx, 4])
            if high_embs is not None and len(high_embs) > det_idx:
                trust = (high_dets[det_idx, 4] - self.det_thresh) / max(1e-6, 1 - self.det_thresh)
                alpha = 0.95 + 0.05 * (1 - trust)
                if self.trackers[trk_idx].get_emb() is None:
                    self.trackers[trk_idx].kf_tracker.emb = high_embs[det_idx]
                else:
                    self.trackers[trk_idx].update_emb(high_embs[det_idx], alpha=alpha)
            matched_tracker_ids.add(trk_idx)

        remaining_high_ids = [int(idx) for idx in unmatched_high]

        if self.use_C_prime and len(cprime_low_ids) > 0:
            self._bprime_diag['frames'] += 1
            lost_trk_indices = [
                int(t)
                for t in unmatched_trackers
                if (
                    self.trackers[int(t)].time_since_update > 0
                    and self.trackers[int(t)].time_since_update <= self.cprime_lost_max_age
                    and self.trackers[int(t)].phd_weight >= self.cprime_track_phd_gate
                )
            ]
            if lost_trk_indices:
                lost_boxes = tracker_boxes[lost_trk_indices]
                active_trk_indices = [i for i in range(len(self.trackers)) if i not in lost_trk_indices]
                active_boxes = tracker_boxes[active_trk_indices] if active_trk_indices else np.empty((0, 5), dtype=float)
                lost_track_phd = np.array([self.trackers[i].phd_weight for i in lost_trk_indices], dtype=float)
                lost_track_dom_share = np.array([getattr(self.trackers[i], 'phd_dominant_share', 0.0) for i in lost_trk_indices], dtype=float)
                promoted = cprime_low_dets.copy()
                original_scores = promoted[:, 4].copy()
                promoted = phd_confidence_boost(
                    promoted,
                    self.phd,
                    self.det_thresh,
                    boost_iou_gate=self.cprime_boost_iou_gate,
                    track_bboxes=lost_boxes,
                )
                promoted_mask = promoted[:, 4] > original_scores
                if np.any(promoted_mask):
                    promoted_dets = promoted[promoted_mask]
                    active_ious = iou_batch(promoted_dets[:, :4], active_boxes[:, :4]) if len(active_boxes) > 0 else np.empty((len(promoted_dets), 0))
                    lost_ious_all = iou_batch(promoted_dets[:, :4], lost_boxes[:, :4]) if len(lost_boxes) > 0 else np.empty((len(promoted_dets), 0))
                    promoted_intensity = np.array([
                        self.phd.evaluate_intensity(phd_convert_bbox_to_z(det[:4])) for det in promoted_dets
                    ], dtype=float)
                    promoted_intensity_norm = self._normalize_phd_intensity(promoted_intensity)
                    if lost_ious_all.shape[1] >= 2:
                        top_sorted = np.sort(lost_ious_all, axis=1)
                        promoted_margins = top_sorted[:, -1] - top_sorted[:, -2]
                    elif lost_ious_all.shape[1] == 1:
                        promoted_margins = lost_ious_all[:, 0].copy()
                    else:
                        promoted_margins = np.zeros(len(promoted_dets), dtype=float)

                    self._bprime_diag['promoted_candidates'] += int(len(promoted_dets))
                    if self.use_any_bprime:
                        keep_promoted_mask = self._select_bprime_group_promotions(active_ious, lost_ious_all)
                    else:
                        keep_promoted_mask = np.ones((len(promoted_dets),), dtype=bool)
                    phd_rescue_mask = self._select_bprime_group_phd_rescue(
                        active_ious,
                        lost_ious_all,
                        promoted_margins,
                        promoted_intensity_norm,
                        lost_track_phd,
                        lost_track_dom_share,
                        keep_promoted_mask,
                    )
                    self._bprime_diag['kept_promoted_candidates'] += int(np.count_nonzero(keep_promoted_mask))
                    self._bprime_diag['filtered_promoted_candidates'] += int(len(promoted_dets) - np.count_nonzero(keep_promoted_mask))
                    self._bprime_diag['phd_rescue_candidates'] += int(np.count_nonzero(phd_rescue_mask))

                    phd_rescue_dets = promoted_dets[phd_rescue_mask] if np.any(phd_rescue_mask) else np.empty((0, promoted_dets.shape[1]), dtype=promoted_dets.dtype)
                    promoted_stage2_dets = promoted_dets[keep_promoted_mask]

                    matched_cp_trk_ids = set()
                    if len(promoted_stage2_dets) > 0:
                        promoted_embs = self.embedder.compute_embedding(img_numpy, promoted_stage2_dets[:, :4], f'{tag}:cprime')
                        lost_trks = tracker_boxes[lost_trk_indices]
                        lost_phd_w = np.array([self.trackers[i].phd_weight for i in lost_trk_indices], dtype=float)
                        lost_emb_cost = None
                        if promoted_embs is not None:
                            lost_embs = np.array([self.trackers[i].get_emb() for i in lost_trk_indices])
                            if lost_embs.size > 0 and promoted_embs.size > 0:
                                lost_emb_cost = promoted_embs.reshape(promoted_embs.shape[0], -1) @ lost_embs.reshape((lost_embs.shape[0], -1)).T

                        matched_cp, _, _, _ = rfs_associate_stage2(
                            promoted_stage2_dets,
                            lost_trks,
                            lost_phd_w,
                            emb_cost=lost_emb_cost,
                            iou_threshold=self.cprime_stage2_threshold,
                            phd_soft_bias=self.cprime_stage2_phd_soft_bias,
                            emb_weight=self.cprime_stage2_emb_weight,
                        )
                        for det_idx, local_trk_idx in matched_cp:
                            det_idx = int(det_idx)
                            local_trk_idx = int(local_trk_idx)
                            global_trk_idx = lost_trk_indices[local_trk_idx]
                            matched_cp_trk_ids.add(global_trk_idx)
                            promoted_det = promoted_stage2_dets[det_idx]
                            self.trackers[global_trk_idx].update(promoted_det, promoted_det[4])
                            if promoted_embs is not None and len(promoted_embs) > det_idx:
                                emb = promoted_embs[det_idx]
                                if self.trackers[global_trk_idx].get_emb() is None:
                                    self.trackers[global_trk_idx].kf_tracker.emb = emb
                                else:
                                    self.trackers[global_trk_idx].update_emb(emb, alpha=0.95)
                            matched_tracker_ids.add(global_trk_idx)

                    if len(phd_rescue_dets) > 0:
                        remaining_lost_trk_indices = [idx for idx in lost_trk_indices if idx not in matched_cp_trk_ids]
                        if remaining_lost_trk_indices:
                            phd_rescue_embs = self.embedder.compute_embedding(img_numpy, phd_rescue_dets[:, :4], f'{tag}:cprime_phd')
                            phd_rescue_lost_trks = tracker_boxes[remaining_lost_trk_indices]
                            phd_rescue_lost_phd_w = np.array([self.trackers[i].phd_weight for i in remaining_lost_trk_indices], dtype=float)
                            phd_rescue_emb_cost = None
                            if phd_rescue_embs is not None:
                                phd_rescue_lost_embs = np.array([self.trackers[i].get_emb() for i in remaining_lost_trk_indices])
                                if phd_rescue_lost_embs.size > 0 and phd_rescue_embs.size > 0:
                                    phd_rescue_emb_cost = phd_rescue_embs.reshape(phd_rescue_embs.shape[0], -1) @ phd_rescue_lost_embs.reshape((phd_rescue_lost_embs.shape[0], -1)).T

                            matched_phd_rescue, _, _, _ = rfs_associate_stage2(
                                phd_rescue_dets,
                                phd_rescue_lost_trks,
                                phd_rescue_lost_phd_w,
                                emb_cost=phd_rescue_emb_cost,
                                iou_threshold=self.bprime_group_phd_stage2_threshold,
                                phd_soft_bias=self.bprime_group_phd_stage2_phd_soft_bias,
                                emb_weight=self.bprime_group_phd_stage2_emb_weight,
                            )
                            for det_idx, local_trk_idx in matched_phd_rescue:
                                det_idx = int(det_idx)
                                local_trk_idx = int(local_trk_idx)
                                global_trk_idx = remaining_lost_trk_indices[local_trk_idx]
                                matched_cp_trk_ids.add(global_trk_idx)
                                self._bprime_diag['phd_rescue_matched'] += 1
                                phd_rescue_det = phd_rescue_dets[det_idx]
                                self.trackers[global_trk_idx].update(phd_rescue_det, phd_rescue_det[4])
                                if phd_rescue_embs is not None and len(phd_rescue_embs) > det_idx:
                                    emb = phd_rescue_embs[det_idx]
                                    if self.trackers[global_trk_idx].get_emb() is None:
                                        self.trackers[global_trk_idx].kf_tracker.emb = emb
                                    else:
                                        self.trackers[global_trk_idx].update_emb(emb, alpha=0.95)
                                matched_tracker_ids.add(global_trk_idx)

                    if matched_cp_trk_ids:
                        unmatched_trackers = np.array([int(t) for t in unmatched_trackers if int(t) not in matched_cp_trk_ids], dtype=int)

        for det_idx in remaining_high_ids:
            det_idx = int(det_idx)
            if high_dets[det_idx, 4] >= self.det_thresh:
                emb = None if high_embs is None else high_embs[det_idx]
                self.trackers.append(MemoTracklet(high_dets[det_idx, :], emb=emb))
        birth_indices = [int(det_idx) for det_idx in remaining_high_ids if high_dets[int(det_idx), 4] >= self.det_thresh]
        if birth_indices:
            birth_dets = high_dets[birth_indices, :4]
            birth_scores = high_dets[birth_indices, 4]
            self.phd.generate_adaptive_birth(birth_dets, birth_scores)
        else:
            self.phd.generate_adaptive_birth(np.array([]), np.array([]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# Compatibility aliases for older experiment scripts.
PHDTracklet = MemoTracklet
PHDTrack = MemoTracker
