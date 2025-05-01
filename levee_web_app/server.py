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
    data = request.json
    response = requests.post(
        "https://aping1100--fs-heaving-api-serve.modal.run/predict",
        json=data
    )
    return jsonify(response.json())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
