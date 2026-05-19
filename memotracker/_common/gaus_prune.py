import numpy as np

def gaus_prune(w, x, P, elim_threshold):
    """
    一对一翻译 MATLAB 版 gaus_prune
    输入
        w : (L,) 权重
        x : (dim, L) 均值
        P : (dim, dim, L) 协方差
        elim_threshold : 剪枝门限
    输出
        w_new : 剪枝后权重
        x_new : 剪枝后均值
        P_new : 剪枝后协方差
    """
    idx = np.where(w > elim_threshold)[0]
    w_new = w[idx]
    x_new = x[:, idx]
    P_new = P[:, :, idx]
    return w_new, x_new, P_new