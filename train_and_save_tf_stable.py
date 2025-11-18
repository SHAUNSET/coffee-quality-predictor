import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv("data/simulated_coffee_data.csv")
X = df[['Temperature', 'Duration']].values.astype(float)
Y = df['Label'].values.astype(float).reshape(-1, 1)

# ---------------------------
# Z-score normalization
# ---------------------------
mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std

# Clip extreme values (optional safeguard)
X_norm = np.clip(X_norm, -5, 5)

print("X_norm min/max:", X_norm.min(), X_norm.max())
print("Y unique values:", np.unique(Y))

# ---------------------------
# Build small Keras model
# ---------------------------
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(3, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    X_norm, Y,
    epochs=500,
    verbose=100
)

# ---------------------------
# Evaluate
# ---------------------------
y_pred_prob = model.predict(X_norm)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(Y, y_pred)
cm = confusion_matrix(Y, y_pred)
report = classification_report(Y, y_pred)

print(f"\nTraining accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# ---------------------------
# Save model and normalization stats
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/tf_coffee_model.keras")  # fixed extension
np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

print("✅ Model and normalization stats saved successfully!")
