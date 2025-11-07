import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Gradient Boosting Classifier Demo")

# Generate some example data
st.write("### Example Dataset")
X = np.random.rand(100, 4)
y = np.random.choice([0, 1], size=100)

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"])
df["Target"] = y
st.write(df.head())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show accuracy
acc = accuracy_score(y_test, y_pred)
st.write("### Model Accuracy:", acc)
