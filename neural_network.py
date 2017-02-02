import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization term

def forward_propagation(params, x):
    ''' Predict probabilities for each class (0 or 1) for each row in x using forward propagation. Uses tanh as the activation function '''
    W1, W2, b1, b2 = params['W1'], params['W2'], params['b1'], params['b2'];
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # Softmax function
    numerator = np.exp(z2)
    a2 = numerator / np.sum(numerator, axis=1, keepdims=True)
    return {'z1': z1, 'z2': z2, 'a1': a1, 'a2': a2}

def back_propagation(fwd_p, params, x, r):
    ''' Get parameter derivatives using back propagation. '''
    W1, W2 = params['W1'], params['W2']
    a1, probs = fwd_p['a1'], fwd_p['a2']
    N = len(x)

    delta3 = probs
    delta3[range(N), r] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)

    # Add regulation terms
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1

    return {'dW1': dW1, 'dW2': dW2, 'db1': db1, 'db2': db2}

def predict(params, x):
    ''' Predict class (0 or 1) for each row in x by choosing the best probability (gained using forward propagation) '''
    return np.argmax(forward_propagation(params, x)['a2'], axis=1)

def predict_confidence_for_class_1(params, x):
    ''' Predict confidence for picking the class 1. 0.95 would mean 95% confidence that a row is classified as 1. 0.1 would mean 90% confidence that a row is classified as 0. Due to heavy penalties on prediction with confidence 100% going wrong, we'll limit the confidence to 0.9 '''
    return np.minimum(0.9, forward_propagation(params, x)['a2'][:,1])

def loss_function(params, x, r):
    ''' Calculate error using cross-entropy loss function '''
    W1, W2 = params['W1'], params['W2']

    N = len(x) # number of rows in x
    probs = forward_propagation(params, x)['a2'] # probabilities for each class. Array of arrays
    # Calculate total error loss
    loss = 1. / N * np.sum( -np.log(probs[range(N), r]) )
    # Add regulatization term to loss (optional)
    loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return loss


def train_params(x, r, hdim, num_passes=100, print_loss=False):
    ''' Train the parameters for neural network.
        x - input data
        r - true labels for x
        hdim - neurons in hidden layer
        num_passes - number of passes through the training data for gradient descent
        print_loss - if True, print the loss every 1000 iterations

        Returns a dictionary where are all the parameters
    '''
    global epsilon
    N = x.shape[0]
    input_dim = x.shape[1]
    output_dim = 2 # we have only 2 classes, 0 and 1
    # Initialize parameters with random values
    W1 = np.random.randn(input_dim, hdim) / np.sqrt(input_dim)
    b1 = np.zeros((1, hdim))
    W2 = np.random.randn(hdim, output_dim) / np.sqrt(hdim)
    b2 = np.zeros((1, output_dim))

    # We return this later
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    errors = np.ones(num_passes) # array of errors (will be plotted in the end to see how error changes over iterations)
    # Gradient descent
    print("\nTRAINING")
    for i in range(0, num_passes):
        fwd_p = forward_propagation(params, x)
        z1, z2, a1, probs = fwd_p['z1'], fwd_p['z2'], fwd_p['a1'], fwd_p['a2']

        back_p = back_propagation(fwd_p, params, x, r)
        dW1, dW2, db1, db2 = back_p['dW1'], back_p['dW2'], back_p['db1'], back_p['db2']

        # Update params
        W1 += -epsilon * dW1
        W2 += -epsilon * dW2
        b1 += -epsilon * db1
        b2 += -epsilon * db2

        params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2':b2}

        loss = loss_function(params, x, r)
        errors[i] = loss
        if print_loss and (i % 10 == 0):
            print("Loss after iteration", i, loss, "    epsilon:",epsilon, "    reg_lambda:", reg_lambda)

    # Plot errors
    plt.plot(errors)
    plt.xlabel("Iteration round")
    plt.ylabel("Error")
    plt.show()

    loss = loss_function(params, x, r)
    print("Loss after iteration", i, loss, "    epsilon:",epsilon, "    reg_lambda:", reg_lambda)
    return params

def main(hdim, eps=0.001, reg_lam=0.001):
    global epsilon
    global reg_lambda
    epsilon = eps
    reg_lambda = reg_lam
    df = pd.read_csv('classification_dataset_training.csv', header=0)
    dims = df.shape
    r = df.ix[:, dims[1]-1].values # last column of dataset is the true labels. Convert it to a NumPy array
    x = df.ix[:, 1:dims[1]-1].values # first column of dataset is just indexing, so ignore it. Convert x to a NumPy array of arrays
    x = x / x.max(axis=0) # normalize x
    x = (x - x.mean(axis=0)) / x.std(axis=0) # normalize x

    params = train_params(x, r, hdim, print_loss=True)

    print("Wrong predictions for training dataset:", sum((predict(params, x) + r)%2),"/",dims[0], "=",sum((predict(params, x) + r)%2)/dims[0],"% error")

    test = pd.read_csv('classification_dataset_testing.csv', header=0)
    x_test = test.ix[:, 1:].values # first column of dataset is just indexing, so ignore it. Convert x to a NumPy array of arrays
    x_test = x_test / x_test.max(axis=0) # normalize x_test
    id_test = test.ix[:, 0] # id column

    preds = predict_confidence_for_class_1(params, x_test)

    output = pd.DataFrame({'ID': id_test,
                           'rating': preds})
    output.to_csv('classification_predictions_test.csv', sep=',', index=False)


if __name__ == "__main__":
    try:
        hdim = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    except ValueError:
        sys.exit("\nArgument should be an integer (denotes the number of neurons in the hidden layer)")
    print("\nNeurons in the hidden layer:", hdim, "\n")
    main(hdim)
