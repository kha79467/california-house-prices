import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# API endpoint for prediction (JSON input)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = regmodel.predict(scaled_data)
    return jsonify(prediction[0])

# Form submission prediction (from HTML form)
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    prediction = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The house price prediction is {prediction}")

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render deployment
    app.run(debug=True, host='0.0.0.0', port=port)
