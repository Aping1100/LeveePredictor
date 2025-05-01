from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import os
import requests
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
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # 只需要 water_levels

    try:
        # 送出請求到 Modal 上的 GPU 模型
        response = requests.post(
            "https://aping1100--fs-heaving-api-serve.modal.run/predict",
            json=data,
            timeout=60
        )
        response.raise_for_status()

        # Modal 已經幫你做完 FS1, FS2 處理
        result = response.json()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Important for cloud hosting
    app.run(host='0.0.0.0', port=port)
