import numpy as np
from src.model import NeuralNetwork
from src.utils import load_data, zscore_normalize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and normalize
X, Y = load_data("data/simulated_coffee_data.csv")
X_norm, mean, std = zscore_normalize(X)

# Training model
nn = NeuralNetwork(lr=0.1)
losses = nn.train(X_norm, Y, epochs=1000, verbose_interval=100)

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("results/training_loss.png")

# Evaluate
y_pred = nn.predict(X_norm)
acc = accuracy_score(Y, y_pred)
print("Training accuracy:", acc)
