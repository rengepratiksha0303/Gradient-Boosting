# app.py
import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# -----------------------------
# Helper function to train/load model
# -----------------------------
MODEL_FILE = "gbt_model.pkl"

def train_and_save_model():
    # Load dataset
    X, y = load_digits(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=23)

    # Train model
    gbt = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        max_features=5,
        random_state=23
    )
    gbt.fit(train_X, train_y)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(gbt, f)

    return gbt, test_X, test_y

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            gbt = pickle.load(f)
        return gbt
    else:
        gbt, test_X, test_y = train_and_save_model()
        return gbt

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Digit Recognizer - Gradient Boosting")

st.write("""
This app predicts handwritten digits (0-9) using a *Gradient Boosting Classifier*.
You can input pixel values (0-16) for each of the 64 features.
""")

# Load model
gbt = load_model()

# Optionally, show model accuracy
X, y = load_digits(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=23)
pred_y = gbt.predict(test_X)
acc = accuracy_score(test_y, pred_y)
st.write(f"Model accuracy on test set: *{acc:.2f}*")

# User input for features
st.subheader("Enter 64 pixel values (0-16):")

# Create 64 input fields (can be improved for UI)
user_input = []
for i in range(64):
    val = st.number_input(f"Pixel {i+1}", min_value=0, max_value=16, value=0)
    user_input.append(val)

# Predict button
if st.button("Predict Digit"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = gbt.predict(input_array)[0]
    st.success(f"Predicted digit: *{prediction}*")
