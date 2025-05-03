from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

app = Flask(__name__)
CORS(app)  # 若前後端分開部署建議加上

# === Define Model ===
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

# === Load Model ===
model = FSHeavingModel()
model.load_state_dict(torch.load("best_model_fs_heaving.pt", map_location=torch.device("cpu")))
model.eval()

# === API Endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        levels = np.array(data["water_levels"]).reshape(1, -1, 1).astype(np.float32)
        levels_tensor = torch.tensor(levels)
        with torch.no_grad():
            pred = model(levels_tensor).numpy().flatten()
        return jsonify({
            "fs1": pred[:40].tolist(),
            "fs2": pred[40:].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import send_from_directory

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


# === Main Run ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
