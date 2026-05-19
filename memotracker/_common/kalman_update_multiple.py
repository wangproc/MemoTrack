import numpy as np
from scipy.linalg import inv, cholesky

def kalman_update_multiple(z, model, m, P):
    # print(f"updata:m.shape={m.shape}, P.shape={P.shape}")
    plength = m.shape[1]
    zlength = z.shape[1]

    qz_update = np.zeros((plength, zlength))
    m_update  = np.zeros((model['x_dim'], plength, zlength))
    P_update  = np.zeros((model['x_dim'], model['x_dim'], plength))

    for idxp in range(plength):
        # 压成二维再计算
        qz_temp, m_temp, P_temp = kalman_update_single(
            z,
            model['H'],
            model['R'],
            m[:, idxp:idxp+1],
            P[:, :, idxp]                  # ← (4,4)
        )
        # 写回
        qz_update[idxp, :]     = qz_temp
        m_update[:, idxp, :]   = m_temp
        P_update[:, :, idxp]   = P_temp

    return qz_update, m_update, P_update


def kalman_update_single(z, H, R, m, P):
    """
    输入
        z : (z_dim, zlength)
        H, R
        m : (x_dim, 1)
        P : (x_dim, x_dim)   ← 二维
    输出
        qz_temp : (zlength,)
        m_temp  : (x_dim, zlength)
        P_temp  : (x_dim, x_dim)
    """
    mu = H @ m
    S  = R + H @ P @ H.T
    Vs = cholesky(S, lower=False)
    det_S = np.prod(np.diag(Vs))**2
    inv_sqrt_S = inv(Vs)
    iS = inv_sqrt_S @ inv_sqrt_S.T
    K  = P @ H.T @ iS

    nu = z - np.tile(mu, (1, z.shape[1]))
    log_lik = (-0.5 * z.shape[0] * np.log(2*np.pi)
               -0.5 * np.log(det_S)
               -0.5 * np.sum(nu * (iS @ nu), axis=0))
    qz_temp = np.exp(log_lik)

    m_temp = np.tile(m, (1, z.shape[1])) + K @ nu
    # 修正：P 已是二维，直接用
    P_temp = (np.eye(P.shape[0]) - K @ H) @ P

    return qz_temp, m_temp, P_temp