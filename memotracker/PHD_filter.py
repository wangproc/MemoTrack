import numpy as np
from copy import deepcopy
from scipy.stats import chi2


from ._common.kalman_predict_multiple import kalman_predict_multiple
from ._common.kalman_update_multiple import kalman_update_multiple
from ._common.gaus_prune import gaus_prune
from ._common.gaus_merge import gaus_merge
from ._common.gaus_cap import gaus_cap
from ._common.gate_meas_gms import gate_meas_gms


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx, cy, h, a]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    a = w / float(h + 1e-6)
    return np.array([cx, cy, h, a])


def convert_z_to_bbox(z):
    """[cx, cy, h, a] -> [x1,y1,x2,y2]"""
    cx, cy, h, a = z[0], z[1], z[2], z[3]
    w = 0 if a <= 0 else a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def convert_state_to_bbox(state_8d):
    """8D state [cx,cy,h,a,vcx,vcy,vh,va] -> [x1,y1,x2,y2]"""
    return convert_z_to_bbox(state_8d[:4])


class PHDFilter:
    """
    GM-PHD filter adapted for MOT bounding box tracking.

    State space: 8D [cx, cy, h, a, vcx, vcy, vh, va]
    Observation space: 4D [cx, cy, h, a]
    Motion model: constant velocity (same as BoostTrack's KalmanFilter)
    """

    def __init__(self, settings, fov_size=(1080, 1920)):
        self.settings = settings
        self.x_dim = settings['x_dim']  # 8
        self.z_dim = settings['z_dim']  # 4

        # Build model dict compatible with phd_base/_common/ functions
        self.model = self._build_model(settings, fov_size)

        # Filter parameters
        self.L_max = settings['L_max']
        self.elim_threshold = settings['elim_threshold']
        self.merge_threshold = settings['merge_threshold']
        self.extract_threshold = settings['extract_threshold']
        self.gate_flag = settings['gate_flag']
        gamma_pval = settings.get('P_G', 0.999)
        self.gamma = chi2.ppf(gamma_pval, self.z_dim)

        # GM-PHD state: weights, means, covariances
        # Initialize with a single negligible component
        self.w_update = np.array([np.spacing(1)])
        self.m_update = np.zeros((self.x_dim, 1))
        self.P_update = np.eye(self.x_dim).reshape(self.x_dim, self.x_dim, 1) * 100.0

        # Birth model (adaptive, generated from unmatched detections)
        self.w_birth = np.array([])
        self.m_birth = np.zeros((self.x_dim, 0))
        self.P_birth = np.zeros((self.x_dim, self.x_dim, 0))

        # Keep a copy of predicted components for intensity evaluation
        self.w_predict = None
        self.m_predict = None
        self.P_predict = None

    def _build_model(self, settings, fov_size):
        """Build model dict with F, H, Q, R matrices matching BoostTrack's KF."""
        x_dim = settings['x_dim']
        z_dim = settings['z_dim']
        dt = 1

        # State transition (constant velocity)
        F = np.eye(x_dim)
        for i in range(4):
            F[i, i + 4] = dt

        # Observation matrix (observe first 4 dims)
        H = np.eye(z_dim, x_dim)

        # Process noise (from ConstantNoise policy)
        Q = np.eye(x_dim)
        Q[4:, 4:] *= 0.01

        # Observation noise (from ConstantNoise policy)
        R = np.diag([1.0, 1.0, 10.0, 0.01])

        # FOV volume for clutter density
        h_fov, w_fov = fov_size
        # Approximate FOV volume in observation space [cx, cy, h, a]
        # cx in [0, w], cy in [0, h], h in [0, h], a in [0, 2]
        fov_volume = w_fov * h_fov * h_fov * 2.0

        model = {
            'x_dim': x_dim,
            'z_dim': z_dim,
            'F': F,
            'H': H,
            'Q': Q,
            'R': R,
            'P_S': settings['P_S'],
            'P_D': settings['P_D'],
            'Q_D': 1.0 - settings['P_D'],
            'lambda_c': settings['lambda_c'],
            'pdf_c': 1.0 / fov_volume,
            'fov_volume': fov_volume,
        }
        return model

    def reset(self):
        """Reset filter state for a new video sequence."""
        self.w_update = np.array([np.spacing(1)])
        self.m_update = np.zeros((self.x_dim, 1))
        self.P_update = np.eye(self.x_dim).reshape(self.x_dim, self.x_dim, 1) * 100.0
        self.w_birth = np.array([])
        self.m_birth = np.zeros((self.x_dim, 0))
        self.P_birth = np.zeros((self.x_dim, self.x_dim, 0))

    def camera_update(self, transform):
        """Apply ECC camera motion transform to all Gaussian components."""
        if transform is None:
            return
        L = self.m_update.shape[1]
        for j in range(L):
            state = self.m_update[:, j]
            bbox = convert_state_to_bbox(state)
            # Apply affine transform to bbox corners
            x1, y1, x2, y2 = bbox
            x1_, y1_, _ = transform @ np.array([x1, y1, 1.0])
            x2_, y2_, _ = transform @ np.array([x2, y2, 1.0])
            w_new = x2_ - x1_
            h_new = y2_ - y1_
            cx_new = x1_ + w_new / 2.0
            cy_new = y1_ + h_new / 2.0
            a_new = w_new / (h_new + 1e-6)
            self.m_update[:4, j] = [cx_new, cy_new, h_new, a_new]

    def predict(self):
        """PHD prediction step: survive existing + add birth components."""
        # Survive existing components
        m_predict, P_predict = kalman_predict_multiple(self.model, self.m_update, self.P_update)
        w_predict = self.model['P_S'] * self.w_update

        # Concatenate birth components
        if self.w_birth.size > 0:
            m_predict = np.concatenate((self.m_birth, m_predict), axis=1)
            P_predict = np.concatenate((self.P_birth, P_predict), axis=2)
            w_predict = np.concatenate((self.w_birth, w_predict))

        self.w_predict = w_predict
        self.m_predict = m_predict
        self.P_predict = P_predict

    def update(self, measurements_z, measurement_scores=None):
        """
        PHD update step.

        Parameters
        ----------
        measurements_z : np.ndarray, shape (z_dim, m)
            Each column is a measurement [cx, cy, h, a].
        measurement_scores : np.ndarray or None, shape (m,)
            Optional detection scores used to down-weight supportive low-score
            measurements during the PHD update.
        """
        if measurements_z.size == 0:
            # No measurements: only missed-detection hypothesis
            self.w_update = self.model['Q_D'] * self.w_predict
            self.m_update = self.m_predict.copy()
            self.P_update = self.P_predict.copy()
            self._prune_merge_cap()
            return

        # Gating
        if self.gate_flag and self.m_predict.shape[1] > 0:
            raw_measurements = measurements_z
            raw_scores = None if measurement_scores is None else np.asarray(measurement_scores, dtype=float).reshape(-1)
            z_gated = gate_meas_gms(raw_measurements, self.gamma, self.model,
                                     self.m_predict, self.P_predict)
            if z_gated.size == 0 or z_gated.shape[1] == 0:
                self.w_update = self.model['Q_D'] * self.w_predict
                self.m_update = self.m_predict.copy()
                self.P_update = self.P_predict.copy()
                self._prune_merge_cap()
                return
            measurements_z = z_gated
            if raw_scores is not None:
                gated_scores = []
                raw_cols = raw_measurements.T
                for ell in range(measurements_z.shape[1]):
                    target = measurements_z[:, ell].reshape(1, -1)
                    mask = np.all(np.isclose(raw_cols, target, atol=1e-8), axis=1)
                    if np.any(mask):
                        gated_scores.append(float(raw_scores[np.argmax(mask)]))
                    else:
                        dist = np.sum((raw_cols - target) ** 2, axis=1)
                        gated_scores.append(float(raw_scores[np.argmin(dist)]))
                measurement_scores = np.asarray(gated_scores, dtype=float)

        m = measurements_z.shape[1]  # number of measurements
        if measurement_scores is None:
            measurement_scores = np.ones(m, dtype=float)
        else:
            measurement_scores = np.asarray(measurement_scores, dtype=float).reshape(-1)
            if measurement_scores.size != m:
                raise ValueError('measurement_scores must match number of measurements')

        # Missed detection hypothesis
        w_update = self.model['Q_D'] * self.w_predict
        m_update = self.m_predict.copy()
        P_update = self.P_predict.copy()

        # Compute Kalman update for all component-measurement pairs
        if m > 0 and self.m_predict.shape[1] > 0:
            qz_temp, m_temp, P_temp = kalman_update_multiple(
                measurements_z, self.model, self.m_predict, self.P_predict)

            # For each measurement, create updated components
            for ell in range(m):
                meas_pd = self._measurement_detection_probability(measurement_scores[ell])
                w_temp = meas_pd * self.w_predict * qz_temp[:, ell]
                denom = self.model['lambda_c'] * self.model['pdf_c'] + np.sum(w_temp)
                if denom > 0:
                    w_temp = w_temp / denom
                w_update = np.concatenate((w_update, w_temp))
                m_update = np.concatenate((m_update, m_temp[:, :, ell]), axis=1)
                P_update = np.concatenate((P_update, P_temp), axis=2)

        self.w_update = w_update
        self.m_update = m_update
        self.P_update = P_update

        self._prune_merge_cap()

    def _measurement_detection_probability(self, score):
        base_pd = float(self.model['P_D'])
        if score is None:
            return base_pd

        det_thresh = float(self.settings.get('det_thresh', 0.6))
        low_thresh = float(self.settings.get('phd_update_low_score_thresh', det_thresh))
        pd_floor = float(self.settings.get('phd_update_pd_low_floor', 0.35))
        pd_power = float(self.settings.get('phd_update_pd_low_power', 1.5))
        score = float(score)

        if score >= det_thresh or det_thresh <= low_thresh:
            return base_pd
        if score <= low_thresh:
            return base_pd * pd_floor

        alpha = (score - low_thresh) / max(det_thresh - low_thresh, 1e-6)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        scale = pd_floor + (1.0 - pd_floor) * (alpha ** pd_power)
        return base_pd * scale

    def _prune_merge_cap(self):
        """Gaussian mixture management: prune, merge, cap."""
        if self.w_update.size == 0:
            return

        self.w_update, self.m_update, self.P_update = gaus_prune(
            self.w_update, self.m_update, self.P_update, self.elim_threshold)

        if self.w_update.size == 0:
            self.w_update = np.array([])
            self.m_update = np.zeros((self.x_dim, 0))
            self.P_update = np.zeros((self.x_dim, self.x_dim, 0))
            return

        self.w_update, self.m_update, self.P_update = gaus_merge(
            self.w_update, self.m_update, self.P_update, self.merge_threshold)

        self.w_update, self.m_update, self.P_update = gaus_cap(
            self.w_update, self.m_update, self.P_update, self.L_max)

    def extract_states(self):
        """
        Extract target states from PHD (weight > threshold).

        Returns
        -------
        states : np.ndarray, shape (N, 8) — 8D state vectors
        bboxes : np.ndarray, shape (N, 4) — [x1, y1, x2, y2]
        weights : np.ndarray, shape (N,) — PHD weights (track confidence)
        covariances : list of np.ndarray — each (8, 8)
        """
        if self.w_update.size == 0:
            return (np.zeros((0, self.x_dim)),
                    np.zeros((0, 4)),
                    np.array([]),
                    [])

        idx = np.where(self.w_update > self.extract_threshold)[0]
        if len(idx) == 0:
            return (np.zeros((0, self.x_dim)),
                    np.zeros((0, 4)),
                    np.array([]),
                    [])

        states = self.m_update[:, idx].T  # (N, 8)
        weights = self.w_update[idx]
        covariances = [self.P_update[:, :, i] for i in idx]

        # Convert to bboxes
        bboxes = np.zeros((len(idx), 4))
        for i, s in enumerate(states):
            bboxes[i] = convert_state_to_bbox(s)

        return states, bboxes, weights, covariances

    def extract_predicted_states(self):
        """
        Extract target states from PHD PREDICTED density (after predict, before update).
        Used in v8 architecture where PHD prediction replaces per-track KF predict.

        Returns same format as extract_states() but from w_predict/m_predict/P_predict.
        """
        if self.w_predict is None or self.w_predict.size == 0:
            return (np.zeros((0, self.x_dim)),
                    np.zeros((0, 4)),
                    np.array([]),
                    [])

        idx = np.where(self.w_predict > self.extract_threshold)[0]
        if len(idx) == 0:
            return (np.zeros((0, self.x_dim)),
                    np.zeros((0, 4)),
                    np.array([]),
                    [])

        states = self.m_predict[:, idx].T  # (N, 8)
        weights = self.w_predict[idx]
        covariances = [self.P_predict[:, :, i] for i in idx]

        bboxes = np.zeros((len(idx), 4))
        for i, s in enumerate(states):
            bboxes[i] = convert_state_to_bbox(s)

        return states, bboxes, weights, covariances

    def evaluate_intensity(self, state_4d):
        """
        Evaluate PHD intensity at a given observation-space point.
        Used for confidence boost (persist intensity).

        Parameters
        ----------
        state_4d : np.ndarray, shape (4,) — [cx, cy, h, a]

        Returns
        -------
        float : PHD intensity value at the given point
        """
        if self.w_update.size == 0:
            return 0.0

        total = 0.0
        H = self.model['H']
        for j in range(len(self.w_update)):
            # Project to observation space
            mu = H @ self.m_update[:, j]
            S = H @ self.P_update[:, :, j] @ H.T + self.model['R']
            # Gaussian pdf
            diff = state_4d - mu
            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
                if det_S <= 0:
                    continue
                exponent = -0.5 * diff @ S_inv @ diff
                pdf = np.exp(exponent) / np.sqrt((2 * np.pi) ** self.z_dim * det_S)
                total += self.w_update[j] * pdf
            except np.linalg.LinAlgError:
                continue
        return total

    def evaluate_birth_intensity(self, state_4d):
        """Evaluate birth intensity at a given point."""
        if self.w_birth.size == 0:
            return 0.0

        total = 0.0
        H = self.model['H']
        for j in range(len(self.w_birth)):
            mu = H @ self.m_birth[:, j]
            S = H @ self.P_birth[:, :, j] @ H.T + self.model['R']
            diff = state_4d - mu
            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
                if det_S <= 0:
                    continue
                exponent = -0.5 * diff @ S_inv @ diff
                pdf = np.exp(exponent) / np.sqrt((2 * np.pi) ** self.z_dim * det_S)
                total += self.w_birth[j] * pdf
            except np.linalg.LinAlgError:
                continue
        return total

    def query_posterior_support(self, state_4d, topk=3, maha_gate=16.0,
                                 ambiguity_floor=0.80, weight_cap=1.10):
        """
        Query posterior GM support for a single track state in observation space.

        Returns a soft evidence estimate built from the top-k gated posterior
        components.  The dominant component id is preserved for Stage 1
        competition, while the blended weight is less brittle than hard nearest-
        component assignment.
        """
        if self.w_update.size == 0:
            return {
                'soft_weight': 0.0,
                'component_id': -1,
                'support_count': 0,
                'dominant_share': 0.0,
            }

        z = np.asarray(state_4d, dtype=float).reshape(self.z_dim)
        H = self.model['H']
        scores = []
        comp_ids = []
        comp_weights = []

        for j in range(len(self.w_update)):
            mu = H @ self.m_update[:, j]
            S = H @ self.P_update[:, :, j] @ H.T + self.model['R']
            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
            except np.linalg.LinAlgError:
                continue

            if det_S <= 0:
                continue

            diff = z - mu
            maha = float(diff @ S_inv @ diff)
            if maha > maha_gate:
                continue

            norm = np.sqrt(max((2 * np.pi) ** self.z_dim * det_S, 1e-12))
            likelihood = np.exp(-0.5 * maha) / norm
            score = float(max(self.w_update[j], 0.0) * likelihood)
            if score <= 0:
                continue

            scores.append(score)
            comp_ids.append(int(j))
            comp_weights.append(float(self.w_update[j]))

        if not scores:
            return {
                'soft_weight': 0.0,
                'component_id': -1,
                'support_count': 0,
                'dominant_share': 0.0,
            }

        scores = np.asarray(scores, dtype=float)
        comp_ids = np.asarray(comp_ids, dtype=int)
        comp_weights = np.clip(np.asarray(comp_weights, dtype=float), 0.0, weight_cap)

        order = np.argsort(scores)[::-1]
        topk = max(int(topk), 1)
        order = order[:topk]

        sel_scores = scores[order]
        sel_ids = comp_ids[order]
        sel_weights = comp_weights[order]
        score_sum = float(sel_scores.sum())
        if score_sum <= 0:
            return {
                'soft_weight': 0.0,
                'component_id': -1,
                'support_count': 0,
                'dominant_share': 0.0,
            }

        responsibilities = sel_scores / score_sum
        dominant_share = float(responsibilities[0])
        blended_weight = float(np.sum(responsibilities * sel_weights))
        ambiguity_factor = float(ambiguity_floor + (1.0 - ambiguity_floor) * dominant_share)
        soft_weight = float(np.clip(blended_weight * ambiguity_factor, 0.0, weight_cap))

        return {
            'soft_weight': soft_weight,
            'component_id': int(sel_ids[0]),
            'support_count': int(len(sel_ids)),
            'dominant_share': dominant_share,
        }

    def get_clutter_density(self):
        """Return clutter density lambda_c * pdf_c."""
        return self.model['lambda_c'] * self.model['pdf_c']

    def generate_adaptive_birth(self, unmatched_dets, det_scores):
        """
        Generate birth components for the next frame from unmatched detections.

        A novelty-aware gate suppresses births that are already well explained by
        the current posterior intensity, reducing duplicate components and
        downstream ID pollution.
        """
        birth_coef = self.settings['birth_weight_coef']
        P_scale = self.settings['birth_P_scale']
        use_novelty = bool(self.settings.get('use_novelty_aware_birth', True))
        support_suppress = float(self.settings.get('birth_support_suppress', 0.28))
        novelty_floor = float(self.settings.get('birth_novelty_floor', 0.30))
        novelty_topk = int(self.settings.get('birth_novelty_topk', 2))
        novelty_maha_gate = float(self.settings.get('birth_novelty_maha_gate', 9.0))
        dominant_gate = float(self.settings.get('birth_novelty_dominant_gate', 0.75))

        if len(unmatched_dets) == 0:
            self.w_birth = np.array([])
            self.m_birth = np.zeros((self.x_dim, 0))
            self.P_birth = np.zeros((self.x_dim, self.x_dim, 0))
            return

        w_list = []
        m_list = []
        P_list = []

        for i in range(len(unmatched_dets)):
            z = convert_bbox_to_z(unmatched_dets[i])
            novelty_factor = 1.0

            if use_novelty:
                support = self.query_posterior_support(
                    z,
                    topk=novelty_topk,
                    maha_gate=novelty_maha_gate,
                    ambiguity_floor=1.0,
                    weight_cap=self.settings.get('phd_weight_cap', 1.10),
                )
                support_weight = float(support['soft_weight'])
                dominant_share = float(support['dominant_share'])

                if support_weight >= support_suppress and dominant_share >= dominant_gate:
                    continue

                if support_suppress > 1e-6:
                    novelty_factor = max(
                        novelty_floor,
                        1.0 - min(support_weight / support_suppress, 1.0)
                    )

            w = det_scores[i] * birth_coef * novelty_factor
            if w < 1e-4:
                continue

            m = np.zeros(self.x_dim)
            m[:4] = z
            P = np.eye(self.x_dim) * P_scale
            P[4:, 4:] *= 10.0

            w_list.append(w)
            m_list.append(m.reshape(-1, 1))
            P_list.append(P)

        if len(w_list) > 0:
            self.w_birth = np.array(w_list)
            self.m_birth = np.hstack(m_list)
            self.P_birth = np.stack(P_list, axis=2)
        else:
            self.w_birth = np.array([])
            self.m_birth = np.zeros((self.x_dim, 0))
            self.P_birth = np.zeros((self.x_dim, self.x_dim, 0))
