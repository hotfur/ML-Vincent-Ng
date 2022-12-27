# HW5 EM clustering
# Vuong Kha Sieu
# Date 7/12/2022
import numpy as np
import sys

def main():
    n, m, iterations = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    start = np.loadtxt(sys.argv[4], dtype=np.float64)
    trans = np.concatenate((np.loadtxt(sys.argv[5], dtype=np.float64), np.zeros(n)[None, :]), axis=0)
    emit = np.concatenate((np.loadtxt(sys.argv[6], dtype=np.float64), np.zeros(m)[None, :]), axis=0)
    data = np.char.strip(a=np.loadtxt(sys.argv[7], dtype=str), chars='o').astype(int)
    seq = data[0]
    alpha = forward(n, start, trans, emit, data[0])
    beta = backward(n, start, trans, emit, data[0])


def calc_epsilon(n, start, trans, emit, seq):
    # Calculate alpha
    alpha = np.empty(shape=(len(seq) + 1, n))
    alpha[-1] = np.zeros(n)
    alpha[-1][-1] = 1
    alpha[0] = start * emit[:, seq[0]]
    for i in range(1, len(seq)):
        alpha[i] = (trans @ alpha[i - 1]) * emit[:, seq[i]]
    # Calculate beta
    beta = np.empty(shape=(len(seq) + 1, n))
    beta[-1] = np.zeros(n)
    beta[-1][-1] = 1
    for i in range(len(seq) - 1, -1, -1):
        beta[i] = (trans @ beta[i + 1]) * emit[:, seq[i]]
    # Calculate gamma
    ab_product = alpha * beta
    gamma = ab_product / np.sum(ab_product, axis=1)[:, None]
    # Calculate epsilon
    epsilon = np.empty(shape=(len(seq), n, n))
    for t in range(0, len(seq)):
        layer = trans * (alpha[t][:, None] * beta[t + 1] * emit[:, seq[t + 1]])
        epsilon[t] = layer/np.sum(layer)
    return gamma, epsilon

def forward(n, start, trans, emit, seq):
    alpha = np.empty(shape=(len(seq)+1, n))
    alpha[-1] = np.zeros(n)
    alpha[-1][-1] = 1
    alpha[0] = start*emit[:, seq[0]]
    for i in range(1, len(seq)):
        alpha[i] = (trans@alpha[i-1])*emit[:, seq[i]]
    return alpha

def backward(n, start, trans, emit, seq):
    beta = np.empty(shape=(len(seq)+1, n))
    beta[-1] = np.zeros(n)
    beta[-1][-1] = 1
    for i in range(len(seq)-1, -1, -1):
        beta[i] = (trans@beta[i+1])*emit[:, seq[i]]
    return beta

if __name__ == '__main__':
    main()