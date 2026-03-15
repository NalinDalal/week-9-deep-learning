
## Neural Network Initialization: Why and How

- **Initialization** is critical for training deep neural networks efficiently.
- Poor initialization can cause slow learning, vanishing/exploding gradients, or failure to break symmetry (all neurons learn the same thing).
- Three main initialization strategies:
  1. **Zeros**: All weights set to zero. Fails to break symmetry; network doesn’t learn.
  2. **Random (large)**: Weights set to large random values. Breaks symmetry, but can cause unstable gradients and slow learning.
  3. **He Initialization**: Weights set to random values scaled by $\sqrt{2/\text{fan-in}}$ (fan-in = number of input units). Works best for ReLU activations.

### Key Takeaways

- Always initialize weights randomly (not zeros).
- Use He initialization for ReLU networks.
- Modularize code for easy swapping/testing of initialization and other components.

---

### Modular init.py Design

- **init.py**: Main entry point, exposes modular functions for initialization, model building, training, and evaluation.
- **initializations.py**: Contains all initialization methods (zeros, random, He, Xavier, etc.).
- **model.py**: Contains the neural network model logic (forward, backward, update, etc.).
- **utils.py**: Helper functions (activation, loss, plotting, etc.).
- **datasets.py**: Data loading and preprocessing.

#### Example init.py Structure

```python
# init.py
from initializations import initialize_parameters
from model import NeuralNetwork
from datasets import load_dataset

def main():
    # Load data
    train_X, train_Y, test_X, test_Y = load_dataset()
    
    # Choose initialization
    init_method = "he"  # or "zeros", "random", etc.
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
```

---

**What you should remember:**
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.