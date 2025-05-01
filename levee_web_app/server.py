from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS

app = Flask(__name__)  # <-- 你可能少了這一行
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'water_levels' not in data:
            return jsonify({'error': 'Missing water_levels in request'}), 400

        response = requests.post(
            "https://aping1100--fs-heaving-api-serve.modal.run/predict",
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return jsonify(response.json())

    except Exception as e:
        # Log and return HTML-safe error message
        print("❌ Flask /predict error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
