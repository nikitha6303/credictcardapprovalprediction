from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Form
@app.route("/predict")
def predict():
    return render_template("index.html")


# Prediction Result
@app.route("/result", methods=["POST"])
def result():

    gender = int(request.form["gender"])
    income_type = int(request.form["income_type"])
    education = int(request.form["education"])
    income = float(request.form["annual_income"])
    family = int(request.form["family_members"])

    features = np.array([[gender,
                          income_type,
                          education,
                          income,
                          family]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        output = "Credit Card Approved"
    else:
        output = "Credit Card Rejected"

    return render_template("result.html", prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv("dataset/credit_card.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")

