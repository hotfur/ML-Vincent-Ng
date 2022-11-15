# HW3 Markov Decision Process
# Vuong Kha Sieu
# Date 15/11/2022

import numpy as np
import sys

def main():
    labels = np.loadtxt(sys.argv[1], dtype=str, max_rows=1)
    train = np.loadtxt(sys.argv[1], dtype=int, skiprows=1)
    test = np.loadtxt(sys.argv[2], dtype=int, skiprows=1)
    alpha, num_iter = float(sys.argv[3]), int(sys.argv[4])
    weights = np.zeros(train.shape[1]-1)
    for i in range(num_iter):
        data = train[np.mod(i, train.shape[0])]
        x = data[None, 0:train.shape[1] - 1]
        xw = x@weights
        output = sigmoid(xw)
        weights += x[0]*alpha*sigmoid(xw, derivative=True)*(data[-1] - output)
        # Printer
        print("After iteration %i: " %(i+1), end='')
        for i in range(len(weights)):
            print("w(%s)=%5.4f" %(labels[i], weights[i]), end=', ')
        print("output=%5.4f" %(output))
    print(f"\nAccuracy on training set ({train.shape[0]} instances): {classification(train, weights):2.1f}%")
    print(f"\nAccuracy on test set ({test.shape[0]} instances): {classification(test, weights):2.1f}%")

def classification(data, weights):
    correct = 0
    for i in range(data.shape[0]):
        row = data[i]
        x = row[None, 0:data.shape[1] - 1]
        if (sigmoid(x @ weights) >= 0.5) == bool(row[-1]):
            correct += 1
    return 100*correct/data.shape[0]

def sigmoid(x, derivative = False):
    f = 1 / (1 + np.e ** (-x))
    if derivative:
        return f * (1 - f)
    return f

if __name__ == "__main__":
    main()