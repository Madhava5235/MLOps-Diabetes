# Flask API for serving the model
#app/app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

# load model
with open('app/model.pkl', 'rb') as f:
    model_data = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Scale features
    scaler = model_data['scaler']
    features_scaled = scaler.transform(features)
    
    # Predict
    model = model_data['model']
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)