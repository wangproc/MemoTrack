import numpy as np
from scipy.linalg import inv, cholesky

def gate_meas_gms(z, gamma, model, m, P):
    # 一对一翻译 MATLAB 版 gate_meas_gms
    valid_idx = []
    zlength = z.shape[1]
    if zlength == 0:
        z_gate = np.array([[]])
        return z_gate
    plength = m.shape[1]

    for j in range(plength):
        Sj = model['R'] + model['H'] @ P[:, :, j] @ model['H'].T
        Vs = cholesky(Sj, lower=False)          # upper triangular
        det_Sj = np.prod(np.diag(Vs))**2
        inv_sqrt_Sj = inv(Vs)
        iSj = inv_sqrt_Sj @ inv_sqrt_Sj.T

        nu = z - model['H'] @ np.tile(m[:, j:j+1], (1, zlength))
        dist = np.sum((inv_sqrt_Sj.T @ nu)**2, axis=0)
        valid_idx = np.union1d(valid_idx, np.where(dist < gamma)[0]).astype(int)

    z_gate = z[:, valid_idx] if valid_idx.size else np.array([[]])
    return z_gate