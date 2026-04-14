
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("credit_card_fraud_10k.csv")

# Remove missing values
data = data.dropna()

# Convert merchant_category to numeric
data["merchant_category"] = data["merchant_category"].astype("category").cat.codes

# Features & label
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# -------------------------
# Home Route
# -------------------------
@app.route("/")
def home():
    return "Federated Fraud Detection API Running"

# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    req = request.json

    sample = [[
        req.get("transaction_id", 1),
        req.get("amount", 100),
        req.get("transaction_hour", 12),
        req.get("merchant_category", 0),
        req.get("foreign_transaction", 0),
        req.get("location_mismatch", 0),
        req.get("device_trust_score", 50),
        req.get("velocity_last_24h", 1),
        req.get("cardholder_age", 30)
    ]]

    prediction = model.predict(sample)[0]

    result = "Fraud Transaction" if prediction == 1 else "Legitimate Transaction"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
