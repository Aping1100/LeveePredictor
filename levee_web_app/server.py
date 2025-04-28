from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import os

app = Flask(__name__)

# ====== Model Definition ======
class FSHeavingModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, output_dim=80, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attn = torch.nn.Linear(hidden_dim, 1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        ctx = torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1)
        return self.fc(ctx)

# ====== Load Model ======
model = FSHeavingModel()

# 🔥 Securely load model relative to server.py location
model_path = os.path.join(os.path.dirname(__file__), 'best_model_fs_heaving.pt')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


@app.route('/')
def index():
    return "Hello from Levee Predictor!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    water_levels = np.array(data['water_levels']).astype(np.float32)

    # Interpolate water levels to 40 points
    wl40 = np.interp(np.linspace(1, 20, 40), np.arange(1, 21), water_levels)
    wl_tensor = torch.tensor(wl40).reshape(1, 40, 1)  # [batch, seq_len, input_dim=1]

    with torch.no_grad():
        output = model(wl_tensor.float())

    output = output.squeeze().numpy()
    fs1 = np.clip(output[:40], 0, 5)
    fs2 = np.clip(output[40:], 0, 5)

    return jsonify({
        'fs1': fs1.tolist(),
        'fs2': fs2.tolist(),
        'water_level': wl40.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Important for cloud hosting
    app.run(host='0.0.0.0', port=port)
