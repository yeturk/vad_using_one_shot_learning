import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

print("\n ************    0. train_lstm_ipad.py is executed by yet :)    ************")

# ======================================================
# 1️⃣ Device Checking
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ======================================================
# 2️⃣ Load Training Features
# ======================================================
train_path = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/training/features/01.npy"
train_features = np.load(train_path)
print(f"📂 Loaded training features: {train_features.shape}")

# ======================================================
# 3️⃣ Sequence Preparation
# ======================================================
sequence_length = 10  # 10 consecutive frame in every sequence
X = []
for i in range(len(train_features) - sequence_length):
    seq = train_features[i:i+sequence_length]
    X.append(seq)

X = np.array(X)
print(f"✅ Prepared sequences: {X.shape}")

# Tensor formatına dönüştür
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset  = TensorDataset(X_tensor, X_tensor)  # Autoencoder: input = output
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ======================================================
# 4️⃣ Define LSTM Autoencoder
# ======================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, seq_len=10):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # hidden → decoder'a giriş olarak ver
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        reconstructed, _ = self.decoder(hidden)
        return reconstructed
    

# Model oluştur
model = LSTMAutoencoder(input_dim=1280, hidden_dim=512, seq_len=sequence_length).to(device)
print(model)

# ======================================================
# 5️⃣ Training Setup
# ======================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 30

# ======================================================
# 6️⃣ Training Loop
# ======================================================
print("\n🚀 Starting training...\n")
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch_X = batch_X.to(device)
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_X)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

print("\n✅ Training completed!")

# ======================================================
# 7️⃣ Save Model
# ======================================================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_ipad.pth")
print("💾 Model saved to models/lstm_ipad.pth")