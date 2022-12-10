# HW5 EM clustering
# Vuong Kha Sieu
# Date 7/12/2022
import numpy as np
import sys


def main():
    train = np.loadtxt(sys.argv[1], dtype=np.float32)
    num_ins = train.shape[0]
    num_gauss, iterations = int(sys.argv[2]), int(sys.argv[3])
    # Initialize probabilities
    prob_arr = np.tile(np.eye(num_gauss), num_ins//num_gauss+1)[:, :num_ins]
    print("M step")
    sumP = np.sum(prob_arr, axis=1)
    mean = (prob_arr @ train)/sumP
    pi = sumP/num_ins
    x_muy = train-mean[:,None]
    variance = (((x_muy*prob_arr)@x_muy.T)/sumP)[:,0]
    print("E step")
    prob = np.exp(-np.square(x_muy)/variance[:,None])/(np.sqrt(2*np.pi)*np.sqrt(variance))


if __name__ == "__main__":
    main()
