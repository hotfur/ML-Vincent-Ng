# HW5 EM clustering
# Vuong Kha Sieu
# Date 7/12/2022
import numpy as np
import sys
import multiprocessing

def main():
    n, m, iterations = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    start = np.loadtxt(sys.argv[4], dtype=np.longdouble)
    trans = np.concatenate((np.loadtxt(sys.argv[5], dtype=np.longdouble), np.zeros(n)[None, :]), axis=0)
    emit = np.concatenate((np.loadtxt(sys.argv[6], dtype=np.longdouble), np.zeros(m)[None, :]), axis=0)
    data = np.char.strip(a=np.loadtxt(sys.argv[7], dtype=str), chars='o').astype(int)-1
    num_seq = data.shape[0]

    pool = multiprocessing.Pool()
    for i in range(iterations):
        args = []
        for seq in data:
            args.append((n, m, start, trans, emit, seq))
        results = pool.starmap(calc_seq, args)

        pi, a_numer, ab_denom, b_numer = np.zeros((num_seq, start.shape[0])), \
            np.zeros((num_seq, trans.shape[0], trans.shape[1])), np.zeros((num_seq, n)), \
            np.zeros((num_seq, emit.shape[0], emit.shape[1]))
        for seq in range(len(results)):
            pi[seq], a_numer[seq], ab_denom[seq], b_numer[seq] = results[seq]
        pi = np.sum(pi, axis=0)
        a_numer = np.sum(a_numer, axis=0)
        ab_denom = np.sum(ab_denom, axis=0)
        b_numer = np.sum(b_numer, axis=0)

        start = pi/num_seq
        trans = a_numer/ab_denom[:,None]
        emit = b_numer/ab_denom[:,None]
    print("fone")
    pool.close()

def calc_seq(n, m, start, trans, emit, seq):
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
        if t==len(seq)-1:
            layer = trans * (alpha[t][:, None] * beta[t + 1])
        else:
            layer = trans * (alpha[t][:, None] * beta[t + 1] * emit[:, seq[t + 1]])
        epsilon[t] = layer/np.sum(layer)
    # Calculate qualities for return
    pi = gamma[0]
    a_numer = np.sum(epsilon, axis=0)
    ab_denom = np.sum(gamma, axis=0)
    # Calculate the indicator mask
    indicator = np.zeros(shape=(len(seq)+1, m))
    for i in range(len(seq)):
        indicator[i, seq[i]] = 1
    b_numer = gamma.T@indicator
    return pi, a_numer, ab_denom, b_numer


if __name__ == '__main__':
    main()