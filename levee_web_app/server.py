from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # 允許跨域請求（如果你的前端是放在別的 domain）

# Proxy endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_proxy():
    modal_url = 'https://aping1100--fs-heaving-api-serve.modal.run/predict'
    try:
        resp = requests.post(modal_url, json=request.json, timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

# Home route (optional, for testing)
@app.route('/')
def home():
    return 'Levee Predictor backend is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
