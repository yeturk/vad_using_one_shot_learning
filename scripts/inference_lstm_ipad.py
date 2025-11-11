import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

print("\n ************    0. inference_lstm_ipad.py is executed by yet :)    ************")

# ======================================================
# 1️⃣ Device Checking
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ======================================================
# 2️⃣ Load Test Features
# ======================================================
# sadece bu kısmı değiştirerek path ayarlayabilirsin (01, 06, 09)
# sample_no hem test_path de hem de save_path de kullanılıyor
sample_no = "09"    
test_path = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/testing/features/"

test_path = test_path + sample_no + ".npy"

test_features = np.load(test_path)
print(f"📂 Loaded test features: {test_features.shape}")

sequence_length = 10
X_test = []
for i in range(len(test_features) - sequence_length):
    seq = test_features[i:i+sequence_length]
    X_test.append(seq)

X_test = np.array(X_test)
print(f"✅ Prepared test sequences: {X_test.shape}")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# ======================================================
# 3️⃣ Define LSTM Autoencoder (same as training)
# ======================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, seq_len=10):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        reconstructed, _ = self.decoder(hidden)
        return reconstructed


# ======================================================
# 4️⃣ Load Model
# ======================================================
model = LSTMAutoencoder(input_dim=1280, hidden_dim=512, seq_len=sequence_length).to(device)
model.load_state_dict(torch.load("/home/yunus/projects/vad_using_one_shot_learning/models/lstm_ipad.pth", 
                                 map_location=device))
model.eval()
print("✅ Model loaded successfully on", device)

# ======================================================
# 5️⃣ Inference - Compute Reconstruction Error
# ======================================================
criterion = nn.MSELoss(reduction='none')
errors = []

print("\n🎯 Starting inference on test data...\n")
with torch.no_grad():
    for seq in tqdm(X_test_tensor):
        seq = seq.unsqueeze(0)
        reconstructed = model(seq)
        loss = criterion(reconstructed, seq).mean().item()
        errors.append(loss)

errors = np.array(errors)
print(f"✅ Inference completed! {len(errors)} sequences processed.")

# ======================================================
# 6️⃣ Save and Visualize Results
# ======================================================
os.makedirs("results", exist_ok=True)

save_path = "/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01_"
save_path = save_path + sample_no + ".npy"

np.save(save_path, errors)
print(f"💾 Saved anomaly scores to {save_path[48:]}")

# Plot
plt.figure(figsize=(12,5))
plt.plot(errors, label="Anomaly Score (Reconstruction Error)")
plt.xlabel("Sequence Index (time)")
plt.ylabel("Error")
plt.title("LSTM Reconstruction Error - IPAD R01")
plt.legend()
plt.grid(True)
plt.show()

print("\n✅ Done! Plot shows higher error regions → anomalies.")