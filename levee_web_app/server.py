from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import os

# === Define model ===
class FSHeavingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=3, batch_first=True)
        self.fc = nn.Linear(128, 80)  # output 80 FS points (40 for fs1 + fs2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# === Init app ===
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# === Load model ===
model = FSHeavingModel()
model.load_state_dict(torch.load("best_model_fs_heaving.pt", map_location=torch.device("cpu")))

model.eval()

# === Serve index.html ===
@app.route('/')
def index():
    return app.send_static_file('index.html')

# === API Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        levels = data.get('water_levels', [])
        if len(levels) != 20:
            return jsonify({'error': '20 water level values required.'}), 400

        # Interpolate to 40 points
        interpolated = []
        for i in range(19):
            interpolated.append(levels[i])
            interpolated.append((levels[i] + levels[i+1]) / 2)
        interpolated.append(levels[-1])

        input_tensor = torch.tensor(interpolated, dtype=torch.float32).view(1, -1, 1)

        with torch.no_grad():
            output = model(input_tensor).view(-1)

        fs1 = output[:40].tolist()
        fs2 = output[40:].tolist() if len(output) >= 80 else fs1  # fallback if only 40 outputs

        return jsonify({'fs1': fs1, 'fs2': fs2})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Start app ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
