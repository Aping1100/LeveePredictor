from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

# === Define your model here ===
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

# === App setup ===
app = Flask(__name__)
CORS(app)

# === Load model ===
model = FSHeavingModel()
model.load_state_dict(torch.load("best_model_fs_heaving.pt", map_location=torch.device("cpu")))
model.eval()

# === /predict endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    water_levels = np.array(data["water_levels"], dtype=np.float32)
    interpolated = []

    for i in range(len(water_levels) - 1):
        interpolated.append(water_levels[i])
        interpolated.append((water_levels[i] + water_levels[i+1]) / 2)
    interpolated.append(water_levels[-1])

    # (1, 40, 1)
    x = torch.tensor(interpolated).view(1, -1, 1)

    with torch.no_grad():
        y_pred = model(x).numpy().flatten()

    # Split output into two sets (FS1_AB and FS2_CD) each with 40 values
    fs1 = y_pred[:40].tolist()
    fs2 = y_pred[40:].tolist() if len(y_pred) == 80 else fs1  # fallback in case only one output

    return jsonify({"fs1": fs1, "fs2": fs2})

# === Required for Railway ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
