# Math.py
import numpy as np
from scipy.optimize import linprog

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
            if maxd > 0:
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


from cvxopt import matrix, spmatrix, solvers
from math import sqrt, log

solvers.options['show_progress'] = False


def ComputeProbas(M, idleness=0): # M is an incidence matrix
    V, E = M.shape
    r = np.arange(V - 1)
    probas = np.zeros((V - 1, V - 1))
    for e in range(E):
        mask = M[1:, e] == 1
        if mask.sum() < 2:
            continue
        cur_vertices = r[mask]
        for i in range(len(cur_vertices)):
            probas[cur_vertices[i], cur_vertices] += 1.0 / (len(cur_vertices) - 1)
    for v in range(V - 1):
        s = M[v + 1].sum()
        if s > 0:
            probas[v] *= (1 - idleness) / M[v + 1].sum()
        probas[v, v] = idleness
    return probas


def ComputeDistances(M):
    V, E = M.shape
    weights = M[0]
    max_weight = 1000000000
    distances = np.ones((V - 1, V - 1)) * max_weight
    r = np.arange(V - 1)
    for e in range(E):
        mask = M[1:, e] == 1
        if mask.sum() < 2:
            continue
        cur_vertices = r[mask]
        for i in range(len(cur_vertices)):
            distances[cur_vertices[i], cur_vertices] = np.clip(distances[cur_vertices[i], cur_vertices], a_min=None, a_max=weights[e])
    for k in range(V - 1):    
        for i in range(V - 1):
            for j in range(V - 1):
                if distances[i, k] < max_weight and distances[k, j] < max_weight:
                    distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
    for v in range(V - 1):
        distances[v, v] = 0
    return distances


def ComputeW(p, q, C):
    obj = -np.concatenate([p, q])
    obj = matrix(obj)
    C = matrix(C)
    n = len(p)
    values = [0] * (n * n * 2)
    rows = [0] * (n * n * 2)
    cols = [0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            index = i * n + j
            values[index * 2] = 1
            values[index * 2 + 1] = 1
            cols[index * 2] = i
            cols[index * 2 + 1] = n + j
            rows[index * 2] = index
            rows[index * 2 + 1] = index
    A = spmatrix(values, rows, cols)
    LP = solvers.lp(obj, A, C, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    return -LP['primal objective']


def CalcGrad(p, q, C):
    obj = -np.concatenate([p, q])
    obj = matrix(obj)
    C = matrix(C)
    n = len(p)
    """
    A = np.zeros((n * n, n * 2))
    for i in range(n):
        for j in range(n):
            A[i * n + j][i] = 1
            A[i * n + j][n + j] = 1
    A = matrix(A)
    """
    values = [0] * (n * n * 2)
    rows = [0] * (n * n * 2)
    cols = [0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            index = i * n + j
            values[index * 2] = 1
            values[index * 2 + 1] = 1
            cols[index * 2] = i
            cols[index * 2 + 1] = n + j
            rows[index * 2] = index
            rows[index * 2 + 1] = index
    A = spmatrix(values, rows, cols)
    LP = solvers.lp(obj, A, C, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    return np.array(LP['x']).reshape(2, n)[0]


def MakeStep(p, q, eta, C):
    exp = np.exp(-eta * CalcGrad(p, q, C))
    return p * exp / (p @ exp)


def StochasticMirrorDescent(C, Q):   # C is the cost matrix (flattened),
    N, n = Q.shape                     # Q is the set of sample distributions
    p = np.zeros(n)
    for i in range(len(Q)):
        p[Q[i] > 0] = 1
    p /= p.sum()
    n_iter = N
    results = np.zeros((n_iter + 1, n))
    results[0] = p
    C_inf = C.max()
    eta = sqrt(2 * log(n)) / (C_inf * sqrt(N))
    
    for k in range(n_iter):
        p = MakeStep(p, Q[k % N], eta, C)
        results[k + 1] = p
    
    return np.mean(results[1:], axis=0)


def hypergraph_curvature(M, idleness=0):
    V, E = M.shape
    probas = ComputeProbas(M, idleness)
    distances = ComputeDistances(M)
    distances_reshaped = distances.reshape(-1)
    r = np.arange(V - 1)
    curvatures = np.zeros(E)
    for e in range(E):
        mask = M[1:, e] == 1
        if mask.sum() < 2: # or edge is deleted
            continue
        cur_probas = probas[mask]
        barycenter = StochasticMirrorDescent(distances_reshaped, cur_probas)
        W_sum = sum(ComputeW(barycenter, cur_probas[i], distances_reshaped) for i in range(len(cur_probas)))
        cur_vertices = r[mask]
        d_sum = None
        for v in cur_vertices:
            cur_d_sum = distances[v, cur_vertices].sum()
            if d_sum is None or cur_d_sum < d_sum:
                d_sum = cur_d_sum
        curvatures[e] = 1 - W_sum / d_sum
    return np.round(curvatures, 3)


def hypergraph_forman(M):
    V, E = M.shape
    d = np.zeros(shape=V)
    m = np.zeros(shape=E)
    res = np.zeros(shape=E)
    for v in range(1, V):
        for e in range(E):
            if M[v, e] != 0.0:
                d[v] += 1
                m[e] += 1
    for e in range(E):
        res[e] = 2 * m[e]
        for v in range(1, V):
            if M[v,e] != 0.0:
                res[e] -= d[v]
    return res
            
# def hypergraph_node_curvature(M, K, index, weighted=False):
#     N = K.shape[0]
#     cur = 0.0
#     for i in range(N):
#         if i != index:
#             cur += K[index][i] * M[index][i]
#     if weighted:
#         d = degree(M, N, index)
#         if d > 0:
#             cur = cur / d
#         else:
#             cur = 0.0
#     return cur

# M = np.array([
#     [1, 1, 1, 0.4],
#     [1, 0, 1, 0],
#     [1, 1, 0, 0],
#     [0, 1, 1, 0]
#     ])
# print(hypergraph_curvature(M, idleness=0.9))
