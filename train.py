import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-8
    return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + epsilon)

def l2_penalty(weights, alpha=0.001):
    return alpha * sum(np.sum(w ** 2) for w in weights)

class MLPBinaryClassifier:
    def __init__(self, input_size, hidden_layers, activations, 
                 dropout_rates=None, learning_rate=0.01, epochs=1000, l2_lambda=0.001):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.dropout_rates = dropout_rates if dropout_rates else [0.0] * len(hidden_layers)
        self.weights = []
        self.biases = []
        self.activation_funcs = []
        self.activation_derivatives = []
        self.train_losses = []
        self.val_losses = []
        self._initialize_network()

    def _initialize_network(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [1]
        
        if len(self.activations) != len(layer_sizes) - 1:
            raise ValueError("# activation functions != # layers")

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

            activation_fn, activation_deriv = ACTIVATIONS[self.activations[i]]
            self.activation_funcs.append(activation_fn)
            self.activation_derivatives.append(activation_deriv)

    def forward(self, X, training=False):
        activations = [X]
        dropout_masks = []
        
        A = X
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activation_funcs[i](Z)

            if training and i < len(self.dropout_rates):
                rate = self.dropout_rates[i]
                if rate > 0:
                    mask = (np.random.rand(*A.shape) > rate).astype(float)
                    A *= mask
                    A /= (1 - rate)
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)
            else:
                dropout_masks.append(None)
                
            activations.append(A)

        return activations, dropout_masks

    def backward(self, activations, y, dropout_masks):
        grads_w = []
        grads_b = []
        
        delta = binary_cross_entropy_derivative(y, activations[-1]) * self.activation_derivatives[-1](activations[-1])

        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, np.dot(activations[i].T, delta))
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivatives[i - 1](activations[i])
                if dropout_masks[i - 1] is not None:
                    delta *= dropout_masks[i - 1]
                    
        return grads_w, grads_b

    def update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, y, X_val=None, y_val=None):
        for epoch in range(self.epochs):
            activations, dropout_masks = self.forward(X, training=True)
            grads_w, grads_b = self.backward(activations, y, dropout_masks)
            self.update_weights(grads_w, grads_b)
            
            train_loss = binary_cross_entropy(y, activations[-1]) + l2_penalty(self.weights, self.l2_lambda)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_preds = self.forward(X_val, training=False)[0][-1]
                val_loss = binary_cross_entropy(y_val, val_preds) + l2_penalty(self.weights, self.l2_lambda)
                self.val_losses.append(val_loss)
            
            if epoch % 100 == 0:
                loss = binary_cross_entropy(y, activations[-1]) + l2_penalty(self.weights, alpha=0.001)
                print(f"Epoch {epoch}, Loss: {loss}")
                if X_val is not None:
                    print(f"Val Loss = {val_loss}")

    def predict(self, X):
        return (self.forward(X)[0][-1] > 0.5).astype(int)

    def plot_loss(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

df = pd.read_csv('data/ckd_clean.csv')

X_df = df.drop(["class"], axis=1)

y = df["class"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train.values)
X_scaled_test = scaler.transform(X_test.values)

model = MLPBinaryClassifier(
    input_size=24,
    hidden_layers=[16, 8],
    activations=["relu", "relu", "sigmoid"],
    dropout_rates=[0.5, 0.5],
    learning_rate=0.01,
    epochs=1000,
    l2_lambda=0.001
)

model.train(X_scaled_train, y_train, X_val=X_scaled_test, y_val=y_test)

with open("mlp_model.pkl", "wb") as f:
    pickle.dump(model, f)