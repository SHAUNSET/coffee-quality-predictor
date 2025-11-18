import numpy as np
from src.utils import load_data, zscore_normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


USE_TF = True

if USE_TF:
    from src.model_tf import NeuralNetworkTF as NeuralNetwork
else:
    from src.model import NeuralNetwork


X, Y = load_data("data/simulated_coffee_data.csv")
X_norm, mean, std = zscore_normalize(X)


nn = NeuralNetwork(lr=0.1)
losses = nn.train(X_norm, Y, epochs=1000, verbose_interval=100)


plt.figure(figsize=(8,5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("results/training_loss.png")
plt.close()


y_pred = nn.predict(X_norm)


cm = confusion_matrix(Y, y_pred)
print("Confusion Matrix:\n", cm)


acc = accuracy_score(Y, y_pred)
prec = precision_score(Y, y_pred)
rec = recall_score(Y, y_pred)
f1 = f1_score(Y, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")


plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

