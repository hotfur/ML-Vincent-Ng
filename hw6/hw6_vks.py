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

    i = 0


def forward(n, start, emit, seq):
    list_alpha = np.empty(shape=(len(seq),n))
    list_alpha[0] = start*emit[:, seq[0]]
    for i in range(1, len(seq)):
        list_alpha[i] = list_alpha[-1]*emit[:]
    return list_alpha

def backward(n, start, emit, seq):
    list_beta = np.empty(shape=(len(seq),n))
    list_beta[0] = np.zeros(n)
    for i in range(1, len(seq)):
        list_beta[i] = list_beta[-1]*emit[:]
    return list_beta

if __name__ == '__main__':
    main()