import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log, exp

def main():
    # load training set
    df_tr = pd.read_csv("datasets/regression_dataset_training.csv")
    y_tr = df_tr['vote']

    df_tr = df_tr.drop(['ID', 'vote'], axis=1) # remove ID and vote columns
    H_tr = df_tr.as_matrix()
    H_tr = np.insert(H_tr, 0, np.ones((H_tr.shape[0])), 1)

    # training
    H_pinv = np.linalg.pinv(H_tr)
    W_hat = np.dot(H_pinv, y_tr)

    # predictions
    y_hat_tr = np.dot(H_tr, W_hat)
    calculate_errors(y_tr, y_hat_tr, "training")

    # load test set
    df_test = pd.read_csv("datasets/regression_dataset_testing.csv")
    y_test = pd.read_csv("datasets/regression_dataset_testing_solution.csv")['vote']

    df_test = df_test.drop(['ID'], axis=1) # remove ID column
    H_test = df_test.as_matrix()
    H_test = np.insert(H_test, 0, np.ones((H_test.shape[0])), 1)

    y_hat_test = np.dot(H_test, W_hat)
    calculate_errors(y_test, y_hat_test, "test")

    bar_plot(W_hat)

def calculate_errors(targets, predictions, set_name):
    """Calculates mean error and RMSE between target values and predictions."""
    n = len(targets)
    rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)
    me = sum(abs(predictions - targets)) / n

    print("Mean error in {:s}: {:.4f}".format(set_name, me))
    print("RMSE in {:s}: {:.4f}".format(set_name, rmse))
    return

def bar_plot(w):
    """Bar plot showing weight coefficients."""
    j = range(len(w))
    width = 1/1.5
    plt.bar(j, w, width, color="blue")
    plt.xlabel("w_j")
    plt.ylabel("value")
    plt.show()
    return


main()
