from flask import Flask, request, jsonify
import numpy as np
import joblib
import util
# Load the model
model=joblib.load("house_price_model.pkl")

app = Flask(__name__)
@app.route('/get_location_names')
def get_location_names():
    respone=jsonify({
        'locations': util.get_location_names()
    })
    return "Welcome to the House Price Prediction API! Use the /predict endpoint to get predictions."


@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.json.get('total_sqft'))
    size = int(request.json.get('size'))
    bath = int(request.json.get('bath'))
    location = request.json.get('location')

    response=jsonify({
        'estimated_price': util.get_EstimatedPrice(location, size, total_sqft, bath)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
 
if __name__ == '__main__':
    app.run(debug=True)
