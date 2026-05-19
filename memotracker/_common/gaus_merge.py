import numpy as np

def gaus_merge(w, x, P, threshold):
    """
    一对一翻译 MATLAB 版 gaus_merge
    输入
        w : (L,) 权重
        x : (dim, L) 均值
        P : (dim, dim, L) 协方差
        threshold : 合并门限
    输出
        w_new : 合并后权重
        x_new : 合并后均值
        P_new : 合并后协方差
    """
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)

    L = len(w)
    x_dim = x.shape[0]
    I = np.arange(L)
    el = 0

    if np.all(w == 0):
        w_new = np.array([])
        x_new = np.array([]).reshape(x_dim, 0)
        P_new = np.array([]).reshape(x_dim, x_dim, 0)
        return w_new, x_new, P_new

    w_new_list = []
    x_new_list = []
    P_new_list = []

    while I.size > 0:
        j = int(np.argmax(w))
        iPt = np.linalg.inv(P[:, :, j])

        Ij = []
        for i in I:
            val = (x[:, i:i+1] - x[:, j:j+1]).T @ iPt @ (x[:, i:i+1] - x[:, j:j+1])
            if val <= threshold:
                Ij.append(i)
        Ij = np.array(Ij, dtype=int)

        w_el = np.sum(w[Ij])
        x_el = wsumvec(w[Ij], x[:, Ij], x_dim)
        P_el = wsummat(w[Ij], P[:, :, Ij], x_dim)

        x_el /= w_el
        P_el /= w_el

        w_new_list.append(w_el)
        x_new_list.append(x_el)
        P_new_list.append(P_el)

        I = np.setdiff1d(I, Ij)
        w[Ij] = -1

    w_new = np.array(w_new_list)
    x_new = np.hstack(x_new_list)
    P_new = np.stack(P_new_list, axis=2)

    return w_new, x_new, P_new


def wsumvec(w, vecstack, xdim):
    """对应 MATLAB 子函数 wsumvec"""
    wmat = np.tile(w.reshape(1, -1), (xdim, 1))
    return np.sum(wmat * vecstack, axis=1, keepdims=True)


def wsummat(w, matstack, xdim):
    """对应 MATLAB 子函数 wsummat"""
    w = w.reshape(1, 1, -1)
    wmat = np.tile(w, (xdim, xdim, 1))
    return np.sum(wmat * matstack, axis=2)