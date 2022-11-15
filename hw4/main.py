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
        output = sigmoid(xw)
        weights += x[0]*alpha*sigmoid(xw, derivative=True)*np.array(data["class"] - output)
        # Printer
        print("After iteration %i: " %(i+1), end='')
        for i in range(len(weights)):
            print("w(%s)=%5.4f" %(data.columns[i], weights[i]), end=', ')
        print("output=%5.4f" %(output))
    print(f"\nAccuracy on training set ({train.shape[0]} instances): {classification(train, weights):2.1f}%")
    print(f"\nAccuracy on test set ({test.shape[0]} instances): {classification(test, weights):2.1f}%")

def classification(data, weights):
    correct = 0
    for i in range(data.shape[0]):
        row = data.iloc[[i]]
        x = np.array(row.iloc[:, 0:data.shape[1] - 1])
        if (sigmoid(x @ weights) >= 0.5) == row["class"].all():
            correct += 1
    return 100*correct/data.shape[0]

def sigmoid(x, derivative = False):
    temp = 1 / (1 + np.e ** (-x))
    if derivative:
        return temp * (1 - temp)
    return temp

if __name__ == "__main__":
    main()