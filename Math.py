# Math.py
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

# функция для подсчета расстояния Вассерштейна
def W1(M, N, m, d, i, j):
    nb_i = []
    nb_j = []
    for nb in range(N):
        if m[i][nb] > 0.0:
            nb_i.append(nb)
        if m[j][nb] > 0.0:
            nb_j.append(nb)
    c = np.zeros(len(nb_i) * len(nb_j))
    A_eq = np.zeros(shape=(len(nb_i) + len(nb_j), len(nb_i) * len(nb_j)))
    b_eq = np.zeros(len(nb_i) + len(nb_j))
    for nb_idx_i in range(len(nb_i)):
        b_eq[nb_idx_i] = m[i][nb_i[nb_idx_i]]
        for nb_idx_j in range(len(nb_j)):
            c[nb_idx_i * len(nb_j) + nb_idx_j] = d[nb_i[nb_idx_i]][nb_j[nb_idx_j]]
            A_eq[nb_idx_i][nb_idx_i * len(nb_j) + nb_idx_j] = 1.0
            A_eq[len(nb_i) + nb_idx_j][nb_idx_i * len(nb_j) + nb_idx_j] = 1.0
    for nb_idx_j in range(len(nb_j)):
        b_eq[len(nb_i) + nb_idx_j] = m[j][nb_j[nb_idx_j]]
    bounds = [(0, 1)] * len(c)
    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds).fun

# степень вершины index 
def degree(M, N, index):
    d = 0.0
    for j in range(N):
        if index != j and M[index][j] != 0:
            #d += M[index][j]
            d += 1
    return d


# расстояния в графе по количеству ребер
def distances(M, N):
    d = np.array(M)
    for i in range(N):
        for j in range(N):
            if d[i][j] != 0:
                d[i][j] = 1
            else:
                d[i][j] = np.inf
        d[i][i] = 0
    for k in range(N):
        for i in range(N):
            for j in range(N):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d

def get_m_x(M, N): #idleness = 0.0
    M_1 = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        d = degree(M, N, i)
        for j in range(N):
            if i == j or M[i][j] == 0:
                M_1[i][j] = 0.0
            else:
                M_1[i][j] = M[i][j] / d
    return M_1

# кривизна Оливье-Риччи обычная
def curvature(M):
    N = len(M[0])
    m = get_m_x(M, N)
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        for j in range(i):
            if M[i][j] != 0:
                K[i][j] = 1 - (W1(M, N, m, d, i, j) / d[i][j])
                K[j][i] = K[i][j]
    return np.around(K, decimals=2)

def get_m_p(M, N, p=None): #constant p if not None
    M_1 = np.zeros(shape = (N, N), dtype = float)
    p_ = (p == None)
    for i in range(N):
        d = degree(M, N, i)
        if p_:
            p = 1 / (d + 1)
        for j in range(N):
            if i == j:
                M_1[i][j] = p
            elif M[i][j] != 0:
                M_1[i][j] = (1 - p) * M[i][j] / d
            else:
                M_1[i][j] = 0
    return M_1

def get_m_lly(M, N):
    M_1 = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        di = degree(M, N, i)
        for j in range(N):
            dj = degree(M, N, j)
            maxd = max(di, dj)
            p = 1/(maxd + 1)
            if i == j:
                M_1[i][j] = p
            elif M[i][j] != 0:
                M_1[i][j] = (1 - p) * M[i][j] / di
            else:
                M_1[i][j] = 0
    return M_1


def curvature_lly(M):
    N = M.shape[0]
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        di = degree(M, N, i)
        for j in range(i):
            dj = degree(M, N, j)
            maxd = max(di, dj)
            m = get_m_p(M, N, p=1/(maxd + 1))
            if M[i][j] != 0:
                K[i][j] = 1 - (W1(M, N, m, d, i, j) / d[i][j])
                K[j][i] = K[i][j]
            K[i][j] *= (maxd + 1)/maxd
            K[j][i] *= (maxd + 1)/maxd
    return np.around(K, decimals=2)

def curvature_with_idleness(M, idleness=None):
    N = M.shape[0]
    m = get_m_p(M, N, p=idleness)
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        for j in range(i):
            if M[i][j] != 0:
                K[i][j] = 1 - (W1(M, N, m, d, i, j) / d[i][j])
                K[j][i] = K[i][j]
    return np.around(K, decimals=2)

def curvature_dir(M):
    N = len(M[0])
    m = get_m_x(M, N)
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        for j in range(N):
            if M[i][j] != 0:
                try:
                    K[i][j] = 1 - (W1(M, N, m, d, i, j) / d[i][j])
                except:
                    pass
                #K[j][i] = K[i][j]
    return np.around(K, decimals=2)


def forman(M):
    N = M.shape[0]
    K = np.zeros(shape = (N, N), dtype = int)
    for i in range(N):
        for j in range(i):
            K[i][j] = 4 - degree(M, N, i) - degree(M, N, j)
            K[j][i] = 4 - degree(M, N, i) - degree(M, N, j)
    return K


def node_curvature(M, K, index, weighted=False):
    N = K.shape[0]
    cur = 0.0
    for i in range(N):
        if i != index:
            cur += K[index][i] * M[index][i]
    if weighted:
        d = degree(M, N, index)
        if d > 0:
            cur = cur / d
        else:
            cur = 0.0
    return cur