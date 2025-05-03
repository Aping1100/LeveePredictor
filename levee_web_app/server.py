from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS if frontend is hosted separately

# Proxy endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_proxy():
    modal_url = 'https://aping1100--fs-heaving-api-serve.modal.run/predict'
    try:
        print("[INFO] Received prediction request")
        print("Data sent:", request.json)

        resp = requests.post(modal_url, json=request.json, timeout=15)
        resp.raise_for_status()
        response_data = resp.json()
        
        print("[INFO] Response from Modal API:", response_data)
        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to connect to Modal API: {e}")
        return jsonify({'error': 'Failed to connect to prediction server. Please try again later.'}), 500

# Home route
@app.route('/')
def home():
    return 'Levee Predictor backend is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
