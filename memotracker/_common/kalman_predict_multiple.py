import numpy as np


def kalman_predict_multiple(model, m, P):
    # print(f"predict:m.shape={m.shape}, P.shape={P.shape}")
    plength = m.shape[1]
    m_predict = np.zeros_like(m)
    P_predict = np.zeros_like(P)

    for idxp in range(plength):
        # 取切片并压成二维
        m_slice = m[:, idxp:idxp + 1]
        P_slice = P[:, :, idxp]  # (4,4)

        m_temp, P_temp = kalman_predict_single(model['F'], model['Q'],
                                               m_slice,
                                               P_slice)
        # 写回
        m_predict[:, idxp:idxp + 1] = m_temp
        P_predict[:, :, idxp] = P_temp  # 二维→三维单帧
    return m_predict, P_predict


def kalman_predict_single(F, Q, m, P):
    """输入/输出均为二维"""
    P = P.squeeze(axis=2) if P.ndim == 3 else P
    m_predict = F @ m
    P_predict = Q + F @ P @ F.T
    return m_predict, P_predict