"""
init.py - Modular Neural Network Initialization and Training
"""
from initializations import initialize_parameters
from model import NeuralNetwork
from datasets import load_dataset

def main():
    # Load data
    train_X, train_Y, test_X, test_Y = load_dataset()

    # Choose initialization method
    init_method = "he"  # Options: "zeros", "random", "he"
    layers_dims = [train_X.shape[0], 10, 5, 1]
    parameters = initialize_parameters(layers_dims, method=init_method)

    # Build and train model
    nn = NeuralNetwork(layers_dims, parameters)
    nn.train(train_X, train_Y, num_iterations=15000, learning_rate=0.01)

    # Evaluate
    print("Train accuracy:", nn.evaluate(train_X, train_Y))
    print("Test accuracy:", nn.evaluate(test_X, test_Y))

if __name__ == "__main__":
    main()
