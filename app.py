from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# =============================
# Load Dataset
# =============================
data = pd.read_csv("credit_card_fraud_10k.csv")

data = data.dropna()
data["merchant_category"] = data["merchant_category"].astype("category").cat.codes

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Drop transaction_id if exists
if "transaction_id" in data.columns:
    data = data.drop(columns=["transaction_id"])

# One-hot encode merchant category
data = pd.get_dummies(data, columns=["merchant_category"])

# Split features and label
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

print("Model training completed")

# =============================
# Home Route
# =============================
@app.route("/")
def home():
    return "Federated Learning Fraud Detection API Running"

# =============================
# Prediction Route
# =============================
@app.route("/predict", methods=["POST"])
def predict():

    input_data = request.json

    df = pd.DataFrame([input_data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=X.columns, fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]

    result = "Fraud" if prediction == 1 else "Not Fraud"

    return jsonify({
        "prediction": result
    })
    from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# load dataset
data = pd.read_csv("credit_card_fraud_10k.csv")
data = data.dropna()

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    amount = data.get("amount", 0)
    hour = data.get("hour", 12)
    
    sample = [[amount, hour, 0,0,0,0,30]]  # dummy features
    
    prediction = model.predict(sample)[0]
    
    return jsonify({
        "fraud_prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


# =============================
# Run App
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
