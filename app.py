from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("house_price_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"])
        ]

        data = np.array([features])

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${prediction*100000:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
    