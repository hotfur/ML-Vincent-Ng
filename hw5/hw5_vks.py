# HW5 EM clustering
# Vuong Kha Sieu
# Date 7/12/2022
import numpy as np
import sys


def main():
    train = np.loadtxt(sys.argv[1], dtype=np.float64)
    num_ins = train.shape[0]
    num_gauss, iterations = int(sys.argv[2]), int(sys.argv[3])
    # Initialize probabilities
    prob_arr = np.tile(np.eye(num_gauss), num_ins // num_gauss + 1)[:, :num_ins]
    for i in range(iterations):
        #M step
        sumP = np.sum(prob_arr, axis=1)
        mean = (prob_arr @ train)/sumP
        pi = sumP/num_ins
        x_muy = train-mean[:, None]
        variance = (((x_muy*prob_arr)@x_muy.T)/sumP).diagonal()
        print("After iteration %d:" % (i+1))
        for k in range(num_gauss):
            print("Gaussian %d: mean = %4.4f, variance = %4.4f, prior = %4.4f" % (k+1, mean[k], variance[k], pi[k]))
        # E step
        prob = np.exp(-np.square(x_muy)/(2*variance)[:, None])/(np.sqrt(2*np.pi*variance)[:, None])
        prob_arr = prob*pi[:,None]/(np.sum(prob*pi[:,None], axis=0))


if __name__ == "__main__":
    main()
