import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, lr=0.1, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, Y, y_hat):
        eps = 1e-8
        return -np.mean(Y * np.log(y_hat + eps) + (1 - Y) * np.log(1 - y_hat + eps))

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = (1/m) * A1.T.dot(dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = (1/m) * X.T.dot(dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        # update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=1000, verbose_interval=100):
        losses = []
        for epoch in range(epochs):
            y_hat, cache = self.forward(X)
            loss = self.compute_loss(Y, y_hat)
            losses.append(loss)
            self.backward(cache, Y)
            if epoch % verbose_interval == 0:
                print(f"Epoch {epoch} | Loss {loss:.4f}")
        return losses

    def predict_proba(self, X):
        y_hat, _ = self.forward(X)
        return y_hat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)
