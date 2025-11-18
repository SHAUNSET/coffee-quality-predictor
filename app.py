import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model_tf import NeuralNetworkTF  # Correct class import
from src.utils import load_data, zscore_normalize
import tensorflow as tf

# Load and normalize data
X_train, Y_train = load_data("data/simulated_coffee_data.csv")
X_norm, mean, std = zscore_normalize(X_train)

# Load trained TensorFlow model
nn_tf = NeuralNetworkTF()
nn_tf.model = tf.keras.models.load_model("models/tf_coffee_model.keras")  # load saved model

st.title("☕ Coffee Roasting Quality Predictor")

st.write("Adjust Temperature and Duration, and see if your coffee batch is Good or Bad!")

# Input sliders
temp = st.slider("Temperature (°C)", int(X_train[:,0].min()), int(X_train[:,0].max()), 200)
dur = st.slider("Duration (minutes)", float(X_train[:,1].min()), float(X_train[:,1].max()), 15.0, step=0.1)

# Prepare input
X_input = np.array([[temp, dur]])
X_input_norm, _, _ = zscore_normalize(X_input, mean, std)

# Prediction
prob = nn_tf.model.predict(X_input_norm)[0][0]
label = "Good ✅" if prob >= 0.5 else "Bad ❌"

st.subheader(f"Prediction: {label} (Probability: {prob:.2f})")

# Plot training data + input
fig, ax = plt.subplots()
good_idx = Y_train[:,0] == 1
bad_idx = Y_train[:,0] == 0

ax.scatter(X_train[good_idx,0], X_train[good_idx,1], color='green', label='Good')
ax.scatter(X_train[bad_idx,0], X_train[bad_idx,1], color='red', label='Bad')
ax.scatter(temp, dur, color='blue', s=150, label='Your Input', marker='X')

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Duration (minutes)")
ax.set_title("Coffee Roasting Data & Prediction")
ax.legend()
st.pyplot(fig)
