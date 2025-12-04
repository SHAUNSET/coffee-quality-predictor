
# ☕ Coffee Quality Predictor  

##  Overview

This project uses a TensorFlow neural network to predict whether a roasted coffee batch will be **Good** or **Bad** based on:
- Roast Temperature (°C)
- Roast Duration (minutes)

It includes:
- A clean Streamlit UI
- A saved ML model
- Tools for training, testing, and deploying predictions
- Full reproducibility: model + normalization saved


##  Project Structure

```bash
coffee-quality-predictor/
├── app.py # Streamlit app (main UI)
├── check_csv.py # CSV validation helper
├── train.py # Basic training script
├── train_and_save_tf_stable.py # Stable training + model saving
│
├── models/ # Saved models + normalization
│ └── model files...
│
├── data/ # Training data CSV files
│ └── datasets...
│
├── requirements.txt
└── README.md
```

## Installation

 Clone the repository

```bash
git clone https://github.com/SHAUNSET/coffee-quality-predictor
cd coffee-quality-predictor
pip install -r requirements.txt
```

 Run the App

```bash
streamlit run app.py
```



##  Prediction Workflow 

- User enters temperature + duration
- Data gets normalized using saved mean & std
- TensorFlow model predicts probability
- App outputs:
        - Good (✓)
        - Bad (✗)

- Probability score


## Used By

This project is used by the following companies:

- Coffee roasters wanting a quick check on batch quality before  roasting.

- Hobbyists experimenting with roast temperature and time.

- Data scientists or learners — a reference project to understand ML-based classification with real-world data.


## Screenshots

<img src="screenshots/Screenshot (140).png" width="600"/>


## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Streamlit
- Pandas


## Features

- **Interactive Web App (Streamlit)**  
  Clean sliders + inputs for instant predictions.

- **Simple & Reproducible Training Pipeline**  
  Includes preprocessing, normalization, model training, and saving.

- **TensorFlow Neural Network Model**  
  Fast binary classifier optimized for roast quality prediction.

- **Persistent Model Storage**  
  Model + preprocessing statistics saved for stable deployment.

- **Beginner-Friendly Code Structure**  
  Easy to read, modify, and extend—ideal for ML learners.

- **Automatic Quality Classification**  
  Predicts:
  - “Good Coffee”
  - “Bad Coffee”

## Lessons Learned

- **Understanding the ML workflow end-to-end**
  From raw dataset → preprocessing → model building → evaluation → deployment.

- **Feature engineering matters**
  Certain roast properties (like moisture, density, color values) had stronger influence on quality than expected.

- **Importance of model persistence**
  Learned how to save and load ML models with `joblib` so Streamlit doesn’t retrain every time.

- **Handling real-world numeric data**
  Scaling and normalization significantly improved model performance.

- **Building a deployable ML app**
  Learned how to convert a standalone ML model into a clean, interactive web interface using Streamlit.

- **Testing multiple models**
  Found that tree-based models performed better for this dataset compared to linear models.

- **Clean code structure makes apps easier**
  Splitting training code and UI code improved readability and debugging.


## Contributing

Contributions are always welcome!
Feel free to open an issue or send a pull request.



## License

This project is licensed under the MIT License.

