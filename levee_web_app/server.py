from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# === Serve index.html from templates ===
@app.route('/')
def home():
    return render_template('index.html')  # ✅ 改這裡：從 templates 回傳 HTML

# === Prediction proxy ===
@app.route('/predict', methods=['POST'])
def predict_proxy():
    modal_url = 'https://aping1100--fs-heaving-api-serve.modal.run/predict'
    try:
        resp = requests.post(modal_url, json=request.json, timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTPError: {http_err}")
        print(f"[ERROR] Response text: {http_err.response.text}")
        return jsonify({'error': f"Modal error {http_err.response.status_code}: {http_err.response.text}"}), 500
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] RequestException: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
