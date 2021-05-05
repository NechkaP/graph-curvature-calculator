import numpy as np
from scipy.optimize import linprog

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

def degree(M, N, index):
    d = 0.0
    for j in range(N):
        if index != j:
            d += M[index][j]
    return d

def get_m_p(M, N, p=0): #clever idleness
    M_1 = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        d = degree(M, N, i)
        for j in range(N):
            if i == j:
                M_1[i][j] = p
            elif M[i][j] != 0:
                M_1[i][j] = (1 - p) * M[i][j] / d
            else:
                M_1[i][j] = 0
    return M_1

def W1(M, N, m, d, i, j, p=0):
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

def curvature(M, p=0, curvature_type='ollivier'):
    N = len(M[0])
    m = get_m_p(M, N, IDLENESS)
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        for j in range(i):
            if M[i][j] != 0:
                K[i][j] = 1 - (W1(M, N, m, d, i, j, IDLENESS) / d[i][j])
                K[j][i] = K[i][j]
    if curvature_type != 'ollivier' and curvature_type != 'idleness':
        return np.zeros(shape=(N, N), dtype=float)
    return np.around(K, decimals=8)

def get_m_x(M, N):
    M_1 = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        d = degree(M, N, i)
        p = 1 / (d + 1)
        for j in range(N):
            if i == j:
                M_1[i][j] = p
            elif M[i][j] != 0:
                M_1[i][j] = (1 - p) * M[i][j] / d
            else:
                M_1[i][j] = 0
    return M_1
def curvature_with_idleness(M, p=0):
    N = M.shape[0]
    m = get_m_x(M, N)
    d = distances(M, N)
    K = np.zeros(shape = (N, N), dtype = float)
    for i in range(N):
        for j in range(i):
            if M[i][j] != 0:
                K[i][j] = 1 - (W1(M, N, m, d, i, j, p) / d[i][j])
                K[j][i] = K[i][j]
    return np.around(K, decimals=8)

def node_curvature(K, index, p=0):
    N = K.shape[0]
    cur = 0.0
    for i in range(N):
        if i != index:
            cur += K[index][i]
    return cur

def node_curvature_weighted(M, K, index, p=0):
    N = K.shape[0]
    d = degree(M, N, index)
    cur = 0.0
    for i in range(N):
        if i != index:
            cur += K[index][i]
    cur = cur * (M[index][i] / d)
    return cur

def graph_curvature():
    # prepare M
    global M
    if WEIGHTS == 'unweighted':
        for i in range(N):
            for j in range(N):
                M[i][j] = int(M[i][j] != 0)
    elif WEIGHTS == 'rounded':
        for i in range(N):
            for j in range(N):
                M[i][j] = int(M[i][j] >= THRESHOLD)
    elif WEIGHTS == 'weighted':
        max_el = 0
        for i in range(N):
            for j in range(N):
                max_el = max(max_el, M[i][j])
        max_el = max(max_el, 1)
        for i in range(N):
            for j in range(N):
                M[i][j] = M[i][j] / max_el
    
    global K
    if CURVATURE == 'idleness':
        if IDLENESS == 0:
            pass
        elif IDLENESS == -1: # clever idleness
            pass
        elif IDLENESS > 1.0 or IDLENESS < 0.0:
            pass
        else:
            pass
    elif CURVATURE == 'forman':
        pass
    elif CURVATURE == 'lly':
        pass
    elif CURVATURE == 'directed':
        pass
    else: # curvature == ollivier
        pass