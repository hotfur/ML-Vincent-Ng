# HW3 Markov Decision Process
# Vuong Kha Sieu
# Date 15/11/2022

import numpy as np
import pandas as pd
import sys

def main():
    train = pd.read_table(sys.argv[1], sep='\t', lineterminator='\n')
    test = pd.read_table(sys.argv[2], sep='\t', lineterminator='\n')
    alpha, num_iter = float(sys.argv[3]), int(sys.argv[4])
    weights = np.zeros(train.shape[1]-1)
    for i in range(num_iter):
        data = train.iloc[[np.mod(i, train.shape[0])]]
        x = np.array(data.iloc[:, 0:train.shape[1]-1])
        xw = x@weights
        weights += x[0]*alpha*sigmoid(xw, derivative=True)*np.array(data["class"] - sigmoid(xw))


def sigmoid(x, derivative = False):
    temp = 1 / (1 + np.e ** (x))
    if derivative:
        return temp * (1 - temp)
    return temp

if __name__ == "__main__":
    main()