import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log, exp

SELECTED_F = ['disappointed','never','delicious','definitely','pretty','bad', 'rating']

def main():
    df_tr = pd.read_csv("datasets/classification_dataset_training.csv")
    df_tr = df_tr.ix[:,1:] # remove ID column
    df_tr = df_tr[SELECTED_F] # select only these features

    N_row = df_tr.shape[0]
    N_col = df_tr.shape[1]
    alpha = 1
    K = 2   # number of classes
    V_abs = N_col - 1 # size of vocabulary

    # split data by class and precompute total observations
    df_0, df_1 = separate_data(df_tr)
    N_0 = count_total_obs(df_0)
    N_1 = count_total_obs(df_1)

    # ML estimates
    r_freq = df_tr.ix[:,N_col-1].value_counts() # frequencies of classes '0' and '1' in all of data
    p_r_hat = r_freq[0] / N_row

    p_0j = calculate_p_hat(df_0, N_0, N_col, V_abs)
    p_1j = calculate_p_hat(df_1, N_1, N_col, V_abs)
    p_hat = [p_0j, p_1j]

    # weights
    w_j = calculate_w_j(p_hat, N_col)
    w_0 = calculate_w_0(p_hat, p_r_hat, N_col)

    #classification
    classification(df_tr, w_j, w_0, N_row, N_col, "training set") #training error

    df_test = pd.read_csv("datasets/classification_dataset_testing.csv")
    test_labels = pd.read_csv("datasets/classification_dataset_testing_solution.csv")
    df_test = attach_labels(df_test, test_labels)
    df_test = df_test[SELECTED_F] # select only SEL_FEATURES
    classification(df_test, w_j, w_0, df_test.shape[0], df_test.shape[1], "testing set") #test error

    w_labels = list(df_tr.columns.values)[:-1]
    w_j_abs = [abs(i) for i in w_j]

    wf = pd.DataFrame(w_j_abs, index = w_labels)
    wf.columns = ['variable']
    sort_wf = wf.sort_values('variable')
    print(sort_wf)

    bar_plot(sort_wf.values)

def separate_data(df):
    """Splits dataframe df into two dataframes by rating value."""
    df_0 = df[df['rating'] == 0]
    df_1 = df[df['rating'] == 1]
    return df_0, df_1

def count_total_obs(df):
    """Counts total number of observations in given dataframe."""
    return df.drop('rating', axis=1).values.sum()

def calculate_p_hat(df, N_tot, N_col, V_abs):
    """Calculates p_hat ML-estimate for given dataframe that ONLY contains data from one class.

    @arg N_tot: total number of observations (sum of all frequencies) in given class
    @arg V_abs: size of vocabulary
    """

    p_hat = []
    for j in range(0,N_col-1):
        numerator = 1 + df.sum(axis=0)[j]
        denominator = V_abs + N_tot
        p_hat.append(numerator / denominator)

    return p_hat

def calculate_w_j(p_hat, N_col):
    w_j = []
    for j in range(0,N_col-1):
        p_0j = p_hat[0][j]
        p_1j = p_hat[1][j]
        w_j.append(log( (p_0j * (1 - p_1j)) / (p_1j * (1 - p_0j)) ))
    return w_j

def calculate_w_0(p_hat, p_r_hat, N_col):
    sum1 = 0
    sum2 = 0
    for j in range(0,N_col-1):
        sum1 += log(1 - p_hat[0][j])
        sum2 += log(1 - p_hat[1][j])
    w_0 = sum1 + log(1 - p_r_hat) - sum2 - log(p_r_hat)
    return w_0

def classification(df, w_j, w_0, N_row, N_col, set_name):
    wrong = 0
    for t in range(N_row):
        g = w_0 + np.dot(w_j, df.ix[t,0:N_col-1])
        if g < 0:
            classify = 1
        else:
            classify = 0

        if df.ix[t,N_col-1] != classify:
            wrong += 1

    print("Error rate in {:s}: {:.4f}".format(set_name, wrong/N_row))
    return

def attach_labels(df, labels):
    df = df.ix[:,1:]
    df['rating'] = labels['rating']
    return df

def bar_plot(w):
    j = range(len(w))
    width = 1/1.5
    plt.bar(j, w, width, color="blue")
    plt.xlabel("w_j")
    plt.ylabel("value")
    plt.show()
    return

main()
