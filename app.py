import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("credit_card_fraud_10k.csv")

# Remove missing fraud labels
data_cleaned = data.dropna(subset=["is_fraud"])

# Encode categorical column
data_encoded = pd.get_dummies(
    data_cleaned,
    columns=["merchant_category"],
    drop_first=True
)

# Features and label
X = data_encoded.drop("is_fraud", axis=1)
y = data_encoded["is_fraud"]

# Save feature order
feature_columns = X.columns.tolist()

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Base Model
# -----------------------------
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train, y_train)

# -----------------------------
# Federated Learning Simulation
# -----------------------------
data_shuffled = data_encoded.sample(frac=1, random_state=42).reset_index(drop=True)

split_size = len(data_shuffled) // 3

client1 = data_shuffled.iloc[:split_size]
client2 = data_shuffled.iloc[split_size:2 * split_size]
client3 = data_shuffled.iloc[2 * split_size:]


def split_xy(client_data):
    X = client_data.drop("is_fraud", axis=1)
    y = client_data["is_fraud"]
    return X, y


X1, y1 = split_xy(client1)
X2, y2 = split_xy(client2)
X3, y3 = split_xy(client3)


def train_local_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


model1 = train_local_model(X1, y1)
model2 = train_local_model(X2, y2)
model3 = train_local_model(X3, y3)

# -----------------------------
# Federated Averaging
# -----------------------------
w1, b1 = model1.coef_, model1.intercept_
w2, b2 = model2.coef_, model2.intercept_
w3, b3 = model3.coef_, model3.intercept_

w_avg = (w1 + w2 + w3) / 3
b_avg = (b1 + b2 + b3) / 3

global_model = LogisticRegression()

global_model.coef_ = w_avg
global_model.intercept_ = b_avg
global_model.classes_ = np.array([0, 1])

# -----------------------------
# API Routes
# -----------------------------

@app.route("/")
def home():
    return "Federated Fraud Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json()

        if "features" not in req:
            return jsonify({"error": "features missing"}), 400

        features = np.array(req["features"]).reshape(1, -1)

        prediction = global_model.predict(features)[0]
        probability = global_model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "result": "Fraud" if prediction == 1 else "Legitimate"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
