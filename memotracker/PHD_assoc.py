import warnings
from copy import deepcopy
from typing import Optional, Tuple, List

import lap
import numpy as np


def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    from default_settings import BoostTrackSettings
    if not BoostTrackSettings['s_sim_corr']:
        return shape_similarity_v1(detects, tracks)
    else:
        return shape_similarity_v2(detects, tracks)

def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))


def shape_similarity_v2(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))


def MhDist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767  # 99% conf interval https://www.mathworks.com/help/stats/chi2inv.html
    mahalanobis_distance = deepcopy(mahalanobis_distance)
    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit
    mahalanobis_distance = limit - mahalanobis_distance

    mahalanobis_distance = np.exp(mahalanobis_distance/softmax_temp) / np.exp(mahalanobis_distance/softmax_temp).sum(0).reshape((1, -1))
    mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
    return mahalanobis_distance


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    return o


def soft_biou_batch(bboxes1, bboxes2):
    """
    Computes soft BIoU between two bboxes in the form [x1,y1,x2,y2]
    BIoU is introduced in https://arxiv.org/pdf/2211.14317
    Soft BIoU is introduced as part of BoostTrack++
    # Author : Vukasin Stanojevic
    # Email  : vukasin.stanojevic@pmf.edu.rs
    """

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    k1 = 0.25
    k2 = 0.5
    b2conf = bboxes2[..., 4]
    b1x1 = bboxes1[..., 0] - (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x1 = bboxes2[..., 0] - (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx1 = np.maximum(b1x1, b2x1)

    b1y1 = bboxes1[..., 1] - (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y1 = bboxes2[..., 1] - (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy1 = np.maximum(b1y1, b2y1)

    b1x2 = bboxes1[..., 2] + (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x2 = bboxes2[..., 2] + (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx2 = np.minimum(b1x2, b2x2)

    b1y2 = bboxes1[..., 3] + (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y2 = bboxes2[..., 3] + (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy2 = np.minimum(b1y2, b2y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (b1x2 - b1x1) * (b1y2 - b1y1)
        + (b2x2 - b2x1) * (b2y2 - b2y1)
        - wh
    )

    return o


def match(cost_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if cost_matrix.size > 0:
        a = (cost_matrix > threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            _, x, y = lap.lapjv(-cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))
    return matched_indices


def linear_assignment(detections: np.ndarray, trackers: np.ndarray,
                      iou_matrix: np.ndarray, cost_matrix: np.ndarray,
                      threshold: float, emb_cost: Optional[np.ndarray] = None):
    if iou_matrix is None and cost_matrix is None:
        raise Exception("Both iou_matrix and cost_matrix are None!")
    if iou_matrix is None:
        iou_matrix = deepcopy(cost_matrix)
    if cost_matrix is None:
        cost_matrix = deepcopy(iou_matrix)
    matched_indices = match(cost_matrix, threshold)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        valid_match = iou_matrix[m[0], m[1]] >= threshold  or (False if emb_cost is None else (iou_matrix[m[0], m[1]] >= threshold / 2 and emb_cost[m[0], m[1]] >= 0.75))
        if valid_match:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix


def associate(
        detections,
        trackers,
        iou_threshold,
        mahalanobis_distance: Optional[np.ndarray] = None,
        track_confidence: Optional[np.ndarray] = None,
        detection_confidence: Optional[np.ndarray] = None,
        emb_cost: Optional[np.ndarray] = None,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
            np.empty((0, 0))
        )
    iou_matrix = iou_batch(detections, trackers)

    cost_matrix = deepcopy(iou_matrix)

    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(detection_confidence.reshape((-1, 1)), track_confidence.reshape((1, -1)))
        conf[iou_matrix < iou_threshold] = 0

        cost_matrix += lambda_iou * conf * iou_batch(detections, trackers)
    else:
        warnings.warn("Detections or tracklet confidence is None and detection-tracklet confidence cannot be computed!")
        conf = None

    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        mahalanobis_distance = MhDist_similarity(mahalanobis_distance)

        cost_matrix += lambda_mhd * mahalanobis_distance
        if conf is not None:
            cost_matrix += lambda_shape * conf * shape_similarity(detections, trackers)

    if emb_cost is not None:
        lambda_emb = (1+lambda_iou+lambda_shape+lambda_mhd) * 1.5
        cost_matrix += lambda_emb * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)


def _bbox_to_cxcywh(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox[:4]
    w = max(float(x2 - x1), 1e-6)
    h = max(float(y2 - y1), 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=float)


def _cxcywh_to_bbox(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    w = max(float(w), 2.0)
    h = max(float(h), 2.0)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


def _extrapolate_lscan_bbox(history: np.ndarray, steps_ahead: int) -> Optional[np.ndarray]:
    if history.ndim != 2 or history.shape[1] < 4 or len(history) < 2:
        return None

    geom = np.asarray([_bbox_to_cxcywh(b) for b in history[:, :4]], dtype=float)
    delta = np.diff(geom, axis=0)
    mean_delta = delta.mean(axis=0)

    mean_delta[2] = np.clip(mean_delta[2], -0.10 * geom[-1, 2], 0.10 * geom[-1, 2])
    mean_delta[3] = np.clip(mean_delta[3], -0.10 * geom[-1, 3], 0.10 * geom[-1, 3])

    ref = geom[-1].copy()
    ref[:2] += max(int(steps_ahead), 1) * mean_delta[:2]
    ref[2:] += max(int(steps_ahead), 1) * mean_delta[2:]
    return _cxcywh_to_bbox(ref[0], ref[1], ref[2], ref[3])


def compute_stage1_track_confidence(
        trackers: np.ndarray,
        base_track_confidence: np.ndarray,
        phd_weights: Optional[np.ndarray] = None,
        track_histories: Optional[List[np.ndarray]] = None,
        track_ages: Optional[np.ndarray] = None,
        track_hit_streaks: Optional[np.ndarray] = None,
        phd_component_ids: Optional[np.ndarray] = None,
        l_scan: int = 4,
        stability_weight: float = 0.20,
        competition_floor: float = 0.45,
        competition_age_decay: float = 0.15,
        competition_hit_bonus: float = 0.05,
        return_details: bool = False):
    """
    Sharpen Stage 1 track priors using short-trajectory stability and
    PHD-component competition.

    The goal is to reduce duplicate/stale tracks competing for the same
    posterior PHD evidence before the first matching stage.
    """
    track_conf = base_track_confidence.ravel().astype(float).copy()
    num_trks = len(track_conf)
    base_conf = track_conf.copy()

    details = {
        'base_track_confidence': base_conf.copy(),
        'phd_factor': np.ones(num_trks, dtype=float),
        'stability': np.ones(num_trks, dtype=float),
        'stability_factor': np.ones(num_trks, dtype=float),
        'competition_factor': np.ones(num_trks, dtype=float),
        'shared_mask': np.zeros(num_trks, dtype=bool),
        'shared_component_groups': 0,
        'shared_component_tracks': 0,
        'max_shared_group_size': 0,
        'mean_shared_group_size': 0.0,
    }

    if num_trks == 0:
        details['final_track_confidence'] = track_conf.copy()
        return (track_conf, details) if return_details else track_conf

    if phd_weights is not None:
        phd_factor = np.clip(np.asarray(phd_weights, dtype=float), 0.1, 1.0)
        details['phd_factor'] = phd_factor
        track_conf *= phd_factor

    ages = np.ones(num_trks, dtype=float) if track_ages is None else np.maximum(np.asarray(track_ages, dtype=float), 1.0)
    hit_streaks = np.zeros(num_trks, dtype=float) if track_hit_streaks is None else np.asarray(track_hit_streaks, dtype=float)
    stability = np.ones(num_trks, dtype=float)

    if track_histories is not None and len(track_histories) == num_trks:
        for j, history in enumerate(track_histories):
            if ages[j] > 1:
                continue
            hist_arr = np.asarray(history, dtype=float)
            if hist_arr.ndim == 1:
                hist_arr = hist_arr.reshape(1, -1)
            hist_arr = hist_arr[-max(int(l_scan), 2):, :4]
            ref_box = _extrapolate_lscan_bbox(hist_arr, steps_ahead=int(ages[j]))
            if ref_box is None:
                continue
            pred_box = trackers[j, :4].reshape(1, 4)
            ref_iou = iou_batch(pred_box, ref_box.reshape(1, 4))
            stability[j] = float(ref_iou[0, 0])

    stability_factor = (1.0 - stability_weight) + stability_weight * np.clip(stability, 0.0, 1.0)
    details['stability'] = stability.copy()
    details['stability_factor'] = stability_factor.copy()
    track_conf *= stability_factor

    if phd_component_ids is not None:
        comp_ids = np.asarray(phd_component_ids, dtype=int)
        valid_comp_ids = comp_ids[comp_ids >= 0]
        if valid_comp_ids.size > 0:
            group_sizes = []
            competition_factor = details['competition_factor']
            shared_mask = details['shared_mask']
            for comp_id in np.unique(valid_comp_ids):
                group = np.where(comp_ids == comp_id)[0]
                if len(group) <= 1:
                    continue

                group_sizes.append(len(group))
                shared_mask[group] = True
                age_factor = np.exp(-competition_age_decay * np.maximum(ages[group] - 1.0, 0.0))
                hit_factor = 1.0 + competition_hit_bonus * np.minimum(hit_streaks[group], 5.0)
                seed = np.maximum(track_conf[group], 1e-6) * age_factor * hit_factor
                max_seed = np.max(seed)
                if max_seed <= 0:
                    continue
                comp_factor = competition_floor + (1.0 - competition_floor) * (seed / max_seed)
                competition_factor[group] = comp_factor
                track_conf[group] *= comp_factor

            if group_sizes:
                details['shared_component_groups'] = int(len(group_sizes))
                details['shared_component_tracks'] = int(np.sum(group_sizes))
                details['max_shared_group_size'] = int(np.max(group_sizes))
                details['mean_shared_group_size'] = float(np.mean(group_sizes))

    details['final_track_confidence'] = track_conf.copy()
    return (track_conf, details) if return_details else track_conf


def phd_confidence_boost(detections: np.ndarray, phd_filter, det_thresh: float,
                         boost_iou_gate: float = 0.0,
                         track_bboxes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    PHD Likelihood-Ratio Confidence Boost (Innovation 1).

    For each detection below det_thresh, evaluate PHD intensity.
    If intensity >= extract_threshold, a significant PHD state exists at that
    location (the filter "believes" a target is there), so the detection is boosted.

    Compared to the LR formulation: LR = intensity/clutter_density is always
    astronomically large (due to fov_volume normalization), so extract_threshold
    provides a more meaningful and stable cutoff.

    The boost_iou_gate optionally restricts boosting to detections near existing
    tracks (IoU >= gate), preventing over-boosting isolated false positives.

    Parameters
    ----------
    detections : np.ndarray, shape (N, 5+) — [x1, y1, x2, y2, score, ...]
    phd_filter : PHDFilter instance (uses previous frame's w_update)
    det_thresh : float — detection confidence threshold
    boost_iou_gate : float — if > 0, only boost if max IoU with any track >= gate
    track_bboxes : np.ndarray, shape (M, 4+) or None — track bboxes for IoU gate

    Returns
    -------
    detections : np.ndarray — same array with potentially boosted scores
    """
    from .PHD_filter import convert_bbox_to_z

    if len(detections) == 0:
        return detections

    # PHD intensity threshold calibrated for this filter's observation noise:
    # R = diag([1, 1, 10, 0.01]).  evaluate_intensity() returns a probability
    # DENSITY (not a weight), whose magnitude is ~ w / sqrt((2π)^4 * det(S)).
    # For R only (P→0): max density = 1/sqrt((2π)^4 * 0.1) ≈ 0.08.
    # For a well-tracked target (w≈0.8, P≈5R):  intensity ≈ 0.002 at exact pos.
    # For a detection 1-sigma off-center:         intensity ≈ 0.0002.
    # Background (no target):                      intensity ≈ 0.
    # extract_threshold (0.5) is a component WEIGHT — incomparable with density.
    # 1e-4 catches detections within ~1σ of a well-tracked GM component.
    intensity_threshold = 1e-4

    for i in range(len(detections)):
        if detections[i, 4] >= det_thresh:
            continue

        # IoU gate: only boost detections that substantially overlap an existing
        # track.  With boost_iou_gate=0.3 this matches Stage 1's IoU threshold,
        # so every boosted detection is guaranteed to match Stage 1 — no risk of
        # creating spurious new tracks from boosted-but-unmatched detections.
        if boost_iou_gate > 0 and track_bboxes is not None and len(track_bboxes) > 0:
            ious = iou_batch(detections[i:i+1, :4], track_bboxes[:, :4])
            if ious.max() < boost_iou_gate:
                continue

        z = convert_bbox_to_z(detections[i, :4])
        intensity = phd_filter.evaluate_intensity(z)

        if intensity >= intensity_threshold:
            # Boost to det_thresh + 0.1 (not det_thresh + 1e-4) so Stage 1's
            # confidence-weighted cost matrix treats this detection with
            # meaningful priority rather than as a lowest-rank candidate.
            detections[i, 4] = max(detections[i, 4], det_thresh + 0.1)

    return detections


def rfs_associate_stage1(
        detections: np.ndarray,
        trackers: np.ndarray,
        phd_weights: Optional[np.ndarray],
        mahalanobis_distance: Optional[np.ndarray] = None,
        track_confidence: Optional[np.ndarray] = None,
        detection_confidence: Optional[np.ndarray] = None,
        emb_cost: Optional[np.ndarray] = None,
        iou_threshold: float = 0.3,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        track_histories: Optional[List[np.ndarray]] = None,
        track_ages: Optional[np.ndarray] = None,
        track_hit_streaks: Optional[np.ndarray] = None,
        phd_component_ids: Optional[np.ndarray] = None,
        traj_l_scan: int = 4,
        traj_stability_weight: float = 0.20,
        competition_floor: float = 0.45,
        competition_age_decay: float = 0.15,
        competition_hit_bonus: float = 0.05,
        track_confidence_precomputed: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1 association: detections <-> active tracks.

    Uses the same cost matrix structure as original BoostTrack associate(),
    but with PHD weights modulating track confidence.

    Parameters
    ----------
    detections : np.ndarray, shape (N, 5+) — [x1,y1,x2,y2,score,...]
    trackers : np.ndarray, shape (M, 5) — [x1,y1,x2,y2,0]
    phd_weights : np.ndarray, shape (M,) or None — PHD weights for tracks
    mahalanobis_distance : np.ndarray or None
    track_confidence : np.ndarray, shape (M,) or None
    detection_confidence : np.ndarray, shape (N,) or None
    emb_cost : np.ndarray or None — ReID similarity matrix (N, M)
    iou_threshold : float
    lambda_iou, lambda_mhd, lambda_shape : float — cost weights

    Returns
    -------
    matches, unmatched_detections, unmatched_trackers, cost_matrix
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
            np.empty((0, 0))
        )

    if track_confidence is not None:
        track_confidence = np.asarray(track_confidence, dtype=float).ravel()
        if not track_confidence_precomputed:
            track_confidence = compute_stage1_track_confidence(
                trackers,
                track_confidence,
                phd_weights=phd_weights,
                track_histories=track_histories,
                track_ages=track_ages,
                track_hit_streaks=track_hit_streaks,
                phd_component_ids=phd_component_ids,
                l_scan=traj_l_scan,
                stability_weight=traj_stability_weight,
                competition_floor=competition_floor,
                competition_age_decay=competition_age_decay,
                competition_hit_bonus=competition_hit_bonus,
            )

    iou_matrix = iou_batch(detections, trackers)
    cost_matrix = deepcopy(iou_matrix)

    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(
            detection_confidence.reshape((-1, 1)),
            track_confidence.reshape((1, -1))
        )
        conf[iou_matrix < iou_threshold] = 0
        cost_matrix += lambda_iou * conf * iou_batch(detections, trackers)

    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        mhd_sim = MhDist_similarity(mahalanobis_distance)
        cost_matrix += lambda_mhd * mhd_sim
        if detection_confidence is not None and track_confidence is not None:
            cost_matrix += lambda_shape * conf * shape_similarity(detections, trackers)

    if emb_cost is not None:
        lambda_emb = (1 + lambda_iou + lambda_shape + lambda_mhd) * 1.5
        cost_matrix += lambda_emb * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)


def rfs_associate_stage2(
        detections: np.ndarray,
        trackers: np.ndarray,
        phd_weights: Optional[np.ndarray] = None,
        emb_cost: Optional[np.ndarray] = None,
        iou_threshold: float = 0.15,
        phd_soft_bias: float = 0.10,
        emb_weight: float = 1.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2 association: remaining detections <-> lost tracks.

    Uses relaxed IoU threshold. PHD weights help prioritize which
    lost tracks are more likely to still exist.

    Parameters
    ----------
    detections : np.ndarray, shape (N, 5+)
    trackers : np.ndarray, shape (M, 5)
    phd_weights : np.ndarray, shape (M,) or None
    emb_cost : np.ndarray or None
    iou_threshold : float — relaxed threshold

    Returns
    -------
    matches, unmatched_detections, unmatched_trackers, cost_matrix
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
            np.empty((0, 0))
        )

    iou_matrix = iou_batch(detections, trackers)
    cost_matrix = deepcopy(iou_matrix)

    # Mild PHD soft-bias is intentionally secondary to the upstream gates.
    if phd_weights is not None:
        phd_boost = np.clip(phd_weights, 0.0, 1.0).reshape((1, -1))
        bias = float(np.clip(phd_soft_bias, 0.0, 1.0))
        cost_matrix = cost_matrix * ((1.0 - bias) + bias * phd_boost)

    if emb_cost is not None:
        cost_matrix += float(emb_weight) * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)


def compute_mh_dist_phd(detections: np.ndarray, track_states: np.ndarray,
                        track_covs: List[np.ndarray], n_dims: int = 4) -> np.ndarray:
    """
    Compute Mahalanobis distance between detections and track states.

    Parameters
    ----------
    detections : np.ndarray, shape (N, 5+) — [x1,y1,x2,y2,score,...]
    track_states : np.ndarray, shape (M, 8) — 8D state vectors
    track_covs : list of np.ndarray — each (8, 8) covariance matrix

    Returns
    -------
    np.ndarray, shape (N, M) — Mahalanobis distances
    """
    from .PHD_filter import convert_bbox_to_z

    if len(detections) == 0 or len(track_states) == 0:
        return np.empty((0, 0))

    N = len(detections)
    M = len(track_states)
    result = np.zeros((N, M))

    # Convert detections to observation space
    z = np.zeros((N, n_dims))
    for i in range(N):
        z[i] = convert_bbox_to_z(detections[i, :4])

    for j in range(M):
        x = track_states[j, :n_dims]
        P = track_covs[j][:n_dims, :n_dims]
        sigma_inv = np.reciprocal(np.diag(P) + 1e-10)
        diff = z - x.reshape((1, -1))
        result[:, j] = (diff ** 2 * sigma_inv.reshape((1, -1))).sum(axis=1)

    return result


def assign_phd_states_to_tracks(
        tracks, phd_bboxes, phd_covs,
        iou_threshold=0.15):
    """
    Assign PHD extracted states to historical tracks via LAP (v8 architecture).

    Parameters
    ----------
    tracks : list of RFSTracklet
    phd_bboxes : np.ndarray, shape (N, 4)
    phd_covs : list of np.ndarray, each (8,8)
    iou_threshold : float
    """
    import lap

    M = len(tracks)
    N = len(phd_bboxes)

    if N == 0:
        for trk in tracks:
            trk.set_velocity_prediction()
        return

    if M == 0:
        return

    track_bboxes = np.array([trk.last_bbox for trk in tracks])  # (M, 4)
    iou_mat = iou_batch(track_bboxes, phd_bboxes)                # (M, N)
    cost_mat = 1.0 - iou_mat

    _, x, _ = lap.lapjv(cost_mat, extend_cost=True)

    for j in range(M):
        phd_idx = x[j]
        if phd_idx >= N or iou_mat[j, phd_idx] < iou_threshold:
            tracks[j].set_velocity_prediction()
        else:
            phd_cov_4x4 = phd_covs[phd_idx][:4, :4]
            tracks[j].set_phd_prediction(phd_bboxes[phd_idx], phd_cov_4x4)


def rfs_associate_stage2_v8(
        detections,
        trackers,
        phd_weights=None,
        vel_pred_boxes=None,
        emb_cost=None,
        iou_threshold=0.15,
        lambda_vel=0.5,
        lambda_phd=0.3):
    """
    Stage 2 association (v8): no hard PHD gate, velocity + PHD soft prior.

    Cost = IoU + lambda_vel * vel_IoU + lambda_phd * phd_weight + 1.5 * ReID
    All lost tracks participate (BTPP-equivalent when lambda_vel=0, lambda_phd=0).

    Parameters
    ----------
    detections : np.ndarray, shape (N, 5+)
    trackers : np.ndarray, shape (M, 5)
    phd_weights : np.ndarray, shape (M,) or None
    vel_pred_boxes : np.ndarray, shape (M, 4) or None
    emb_cost : np.ndarray or None
    iou_threshold : float
    lambda_vel, lambda_phd : float
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.arange(len(trackers)),
                np.empty((0, 0)))

    iou_matrix = iou_batch(detections, trackers)
    cost_matrix = deepcopy(iou_matrix)

    if vel_pred_boxes is not None and len(vel_pred_boxes) > 0:
        vel_iou = iou_batch(detections[:, :4], vel_pred_boxes[:, :4])
        cost_matrix += lambda_vel * vel_iou

    if phd_weights is not None:
        phd_bonus = np.clip(phd_weights, 0.0, 1.0).reshape((1, -1))
        cost_matrix += lambda_phd * phd_bonus

    if emb_cost is not None:
        cost_matrix += 1.5 * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix,
                             iou_threshold, emb_cost)
