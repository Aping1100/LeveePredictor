from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS if frontend is hosted separately

# Proxy endpoint for prediction
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict_proxy():
    modal_url = 'https://aping1100--fs-heaving-api-serve.modal.run/predict'
    try:
        resp = requests.post(modal_url, json=request.json, timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.HTTPError as http_err:
        # ➕ 印出 status code 和 response 內容
        print(f"[ERROR] HTTPError: {http_err}")
        print(f"[ERROR] Response text: {http_err.response.text}")
        return jsonify({'error': f"Modal error {http_err.response.status_code}: {http_err.response.text}"}), 500
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] RequestException: {e}")
        return jsonify({'error': str(e)}), 500

# Home route
@app.route('/')
def home():
    return 'Levee Predictor backend is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
