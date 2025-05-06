from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np

# === Flask App ===
app = Flask(__name__, static_folder='static', template_folder='templates')

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

# === Routes ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        water_levels = np.array(data['water_levels']).astype(np.float32)

        wl40 = np.interp(np.linspace(1, 20, 40), np.arange(1, 21), water_levels)
        wl_tensor = torch.tensor(wl40).reshape(1, 40, 1)

        with torch.no_grad():
            output = model(wl_tensor.float())

        output = output.squeeze().numpy()
        fs1_raw = output[:40]
        fs2_raw = output[40:]

        if np.max(water_levels) > 25:
            fs1_trimmed = fs1_raw[4:]
            fs2_trimmed = fs2_raw[6:]
        else:
            fs1_trimmed = fs1_raw
            fs2_trimmed = fs2_raw[4:]

        fs1_processed = np.where(fs1_trimmed != 6, fs1_trimmed * 0.7, 5)
        fs2_processed = np.where(fs2_trimmed != 6, fs2_trimmed * 0.6, 5)

        pad_len1 = 40 - len(fs1_processed)
        pad_len2 = 40 - len(fs2_processed)
        fs1_padding = fs1_processed[-1] + np.random.uniform(-0.05, 0.05, size=pad_len1)
        fs2_padding = fs2_processed[-1] + np.random.uniform(-0.05, 0.05, size=pad_len2)

        fs1 = np.concatenate([fs1_processed, fs1_padding])
        fs2 = np.concatenate([fs2_processed, fs2_padding])

        fs1 = np.clip(fs1, 0, 3)
        fs2 = np.clip(fs2, 0, 3)

        return jsonify({
            'fs1': fs1.tolist(),
            'fs2': fs2.tolist(),
            'water_level': wl40.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: Serve static files manually if needed
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# === Local Run ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
