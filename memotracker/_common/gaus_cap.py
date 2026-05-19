import numpy as np

def gaus_cap(w, x, P, max_number):
    """
    一对一翻译 MATLAB 版 gaus_cap
    输入
        w : (L,) 权重向量
        x : (dim, L) 状态均值矩阵
        P : (dim, dim, L) 协方差张量
        max_number : 保留的最大分量数
    输出
        w_new : 裁剪后权重
        x_new : 裁剪后均值
        P_new : 裁剪后协方差
    """
    if len(w) > max_number:
        # 按权重降序排序
        idx = np.argsort(w)[::-1]
        idx_keep = idx[:max_number]

        w_new = w[idx_keep]
        # 保持总权重不变
        w_new = w_new * (np.sum(w) / np.sum(w_new))

        x_new = x[:, idx_keep]
        P_new = P[:, :, idx_keep]
    else:
        x_new = x
        P_new = P
        w_new = w

    return w_new, x_new, P_new