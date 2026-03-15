"""
utils.py - Utility functions for neural networks
Implements forward, backward propagation, parameter update, and prediction.
"""
import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def compute_loss(a3, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3 + 1e-8), Y) + np.multiply(-np.log(1 - a3 + 1e-8), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    return np.squeeze(cost)

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3, X)
    return A3, cache

def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3, X) = cache

    dZ3 = A3 - Y
    dW3 = (1./m) * np.dot(dZ3, A2.T)
    db3 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.array(dA2, copy=True)
    dZ2[A2 <= 0] = 0
    dW2 = (1./m) * np.dot(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.array(dA1, copy=True)
    dZ1[A1 <= 0] = 0
    dW1 = (1./m) * np.dot(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads

def update_parameters(parameters, grads, learning_rate):
    for l in range(1, 4):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

def predict(X, Y, parameters):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)
    return predictions
