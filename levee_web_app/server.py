from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
import os

app = Flask(__name__, static_folder="static", static_url_path="")

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
model.load_state_dict(torch.load("best_model_fs_heaving.pt", map_location=torch.device("cpu")))
model.eval()
from flask import Flask, request, jsonify, render_template

# ====== Serve index.html ======
@app.route("/")
def home():
    return render_template("index.html")

# ====== Predict API ======
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

# ====== Run Locally ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
