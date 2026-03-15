"""
model.py - Neural network model class
"""
import numpy as np
from utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation, update_parameters, predict

class NeuralNetwork:
    def __init__(self, layers_dims, parameters):
        self.layers_dims = layers_dims
        self.parameters = parameters

    def train(self, X, Y, num_iterations=15000, learning_rate=0.01, print_cost=True):
        costs = []
        for i in range(num_iterations):
            a3, cache = forward_propagation(X, self.parameters)
            cost = compute_loss(a3, Y)
            grads = backward_propagation(X, Y, cache)
            self.parameters = update_parameters(self.parameters, grads, learning_rate)
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        return costs

    def evaluate(self, X, Y):
        predictions = predict(X, Y, self.parameters)
        accuracy = np.mean(predictions == Y)
        return accuracy
