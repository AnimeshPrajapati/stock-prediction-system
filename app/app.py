from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
scaler = joblib.load("model/scaler.pkl")
model = load_model("model/lstm_model.h5")

def prepare_data(df, steps=60):
    data = df[['Close']]
    scaled = scaler.transform(data)
    X = [scaled[-steps:]]
    return np.array(X).reshape((1, steps, 1))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        ticker = request.form["ticker"]
        df = yf.download(ticker, period="6mo")
        if not df.empty:
            X_input = prepare_data(df)
            pred = model.predict(X_input)
            prediction = scaler.inverse_transform(pred)[0][0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
