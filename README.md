Coffee Roasting Quality Predictor
Predict whether a coffee batch will be Good or Bad based on roasting temperature and duration.

Features
•	☕ Simple, intuitive Streamlit UI.
•	⚡ Powered by a TensorFlow neural network.
•	📊 Real-time predictions with probability scores.
•	🛠️ Includes data normalization and trained model for immediate use.

Why this project?
Coffee roasting is an art and a science. This project helps predict coffee batch quality, combining machine learning and practical roasting parameters to give you consistent results.
Getting Started

Requirements
•	Python 3.10+
•	Packages:
 	pip install -r requirements.txt


Folder Structure
coffee-roasting/
│
├─ data/                   # CSV datasets
├─ models/                 # Saved TensorFlow model
├─ src/
│  ├─ model_tf.py          # TensorFlow model class
│  └─ utils.py             # Data loading & normalization
├─ results/                # Training loss plots
├─ app.py                  # Streamlit UI
└─ train_and_save_tf_stable.py # Training script

Usage
1.	Launch the Streamlit app:
 	streamlit run app.py
2.	Enter Temperature (°C) and Duration (minutes).
3.	Get the prediction: Good ✅ or Bad ❌, with probability score.

Training Your Own Model
python train_and_save_tf_stable.py
•	Trains the neural network on simulated_coffee_data.csv.
•	Saves trained model in models/tf_coffee_model.keras.
•	Normalization stats saved in models/normalization.npz.

Results
•	Loss curve: results/training_loss.png.
•	Training accuracy and classification metrics displayed after training.
