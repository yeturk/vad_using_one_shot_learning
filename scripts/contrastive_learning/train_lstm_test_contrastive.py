import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# =========================
# 1️⃣ Configuration
# =========================
class Config:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    data_path = "../data/IPAD_dataset/R03/training/features/R03-01.npy"
    sequence_length = 10
    fps = 30
    
    # Model
    input_dim = 1280
    hidden_dim = 512
    projection_dim = 256
    text_encoder_name = "all-MiniLM-L6-v2"
    text_dim = 384
    
    # Training
    batch_size = 16
    epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Fine-tuning
    finetune_lstm = True  # ✅ LSTM'i de eğit
    lstm_lr_multiplier = 0.1  # LSTM için daha düşük lr
    
    # Loss
    temperature = 0.07
    use_hard_negatives = True
    
    # Data split
    train_ratio = 0.8
    val_ratio = 0.2
    random_seed = 42
    
    # Paths
    model_dir = Path("../models")
    model_dir.mkdir(parents=True, exist_ok=True)

config = Config()
print(f"Device: {config.device}")
print(f"Fine-tune LSTM: {config.finetune_lstm}")

# =========================
# 2️⃣ Enhanced Text Annotations
# =========================
# Her saniye için daha detaylı text (22 saniye → 22 text)
texts = [
    "forklift approaching the cylindrical load on the left",
    "forklift aligned directly in front of the load",
    "forklift lowering its forks toward the load",
    "forklift sliding forks under the load",
    "forklift lifting the cylindrical load upward",
    "forklift holding the raised load",
    "forklift moving backward with the lifted load",
    "forklift transporting the elevated load backward",
    "forklift turning slightly while carrying the load",
    "forklift holding the load at maximum height",
    "forklift keeping the fully raised load stable",
    "forklift holding the elevated load near the top of the frame",
    "forklift adjusting position while keeping the load raised",
    "forklift holding the suspended load still",
    "forklift beginning to lower the load",
    "forklift lowering the load halfway",
    "forklift lowering the load near the ground",
    "forklift placing the load on the ground",
    "forklift retracting its forks",
    "forklift moving backward away from the load",
    "forklift reversing farther from the load",
    "forklift leaving the frame after placing the load"
]

# =========================
# 3️⃣ Load and Prepare Data
# =========================
features = np.load(config.data_path)
print(f"Loaded features: {features.shape}")

# Create sequences
sequences = []
frame_indices = []  # Her sequence'in başlangıç frame'i

for i in range(len(features) - config.sequence_length):
    sequences.append(features[i:i+config.sequence_length])
    frame_indices.append(i)

sequences = np.array(sequences)
frame_indices = np.array(frame_indices)

print(f"Total sequences: {len(sequences)}")

# =========================
# 4️⃣ Improved Dataset with Better Alignment
# =========================
class VideoTextDataset(Dataset):
    def __init__(self, video_seqs, frame_indices, texts, fps=30):
        self.video_seqs = video_seqs
        self.frame_indices = frame_indices
        self.texts = texts
        self.fps = fps
        
    def __len__(self):
        return len(self.video_seqs)
    
    def __getitem__(self, idx):
        # Sequence'in ortasındaki frame'e göre text seç
        center_frame = self.frame_indices[idx] + config.sequence_length // 2
        second_idx = center_frame // self.fps
        
        # Text boundary check
        text_idx = min(second_idx, len(self.texts) - 1)
        text = self.texts[text_idx]
        
        return torch.tensor(self.video_seqs[idx], dtype=torch.float32), text, idx

# =========================
# 5️⃣ Train/Val Split
# =========================
train_indices, val_indices = train_test_split(
    np.arange(len(sequences)),
    train_size=config.train_ratio,
    random_state=config.random_seed,
    shuffle=True
)

train_sequences = sequences[train_indices]
train_frame_indices = frame_indices[train_indices]

val_sequences = sequences[val_indices]
val_frame_indices = frame_indices[val_indices]

print(f"Train sequences: {len(train_sequences)}")
print(f"Val sequences: {len(val_sequences)}")

# Create datasets
train_dataset = VideoTextDataset(train_sequences, train_frame_indices, texts, config.fps)
val_dataset = VideoTextDataset(val_sequences, val_frame_indices, texts, config.fps)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# =========================
# 6️⃣ LSTM Encoder
# =========================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return h[-1]  # (batch, hidden_dim)

video_encoder = LSTMEncoder(config.input_dim, config.hidden_dim).to(config.device)

# Load pre-trained weights if available
try:
    video_encoder.load_state_dict(
        torch.load("models/lstm_ipad.pth", map_location=config.device),
        strict=False
    )
    print("✅ Loaded pre-trained LSTM weights")
except:
    print("⚠️ No pre-trained LSTM found, training from scratch")

# Set training mode based on config
if config.finetune_lstm:
    video_encoder.train()
    print("✅ LSTM in training mode (will be fine-tuned)")
else:
    video_encoder.eval()
    print("🔒 LSTM frozen (only projection heads will be trained)")

# =========================
# 7️⃣ Text Encoder
# =========================
text_encoder = SentenceTransformer(
    config.text_encoder_name,
    device=config.device
)

# =========================
# 8️⃣ Projection Heads
# =========================
proj_video = nn.Linear(config.hidden_dim, config.projection_dim).to(config.device)
proj_text = nn.Linear(config.text_dim, config.projection_dim).to(config.device)

# =========================
# 9️⃣ Symmetric Contrastive Loss
# =========================
def contrastive_loss(v_emb, t_emb, temperature=0.07):
    """
    Symmetric contrastive loss (CLIP-style)
    
    Args:
        v_emb: Video embeddings (batch, dim) - normalized
        t_emb: Text embeddings (batch, dim) - normalized
        temperature: Temperature parameter
    
    Returns:
        loss: Symmetric contrastive loss
    """
    # Compute similarity matrix
    logits = torch.matmul(v_emb, t_emb.T) / temperature  # (batch, batch)
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(v_emb)).to(v_emb.device)
    
    # Video-to-text loss
    loss_v2t = F.cross_entropy(logits, labels)
    
    # Text-to-video loss
    loss_t2v = F.cross_entropy(logits.T, labels)
    
    # Symmetric loss
    loss = (loss_v2t + loss_t2v) / 2
    
    return loss, loss_v2t, loss_t2v

# =========================
# 🔟 Hard Negative Mining (Optional)
# =========================
def hard_negative_loss(v_emb, t_emb, temperature=0.07, hard_neg_ratio=0.3):
    """
    Contrastive loss with hard negative mining
    """
    logits = torch.matmul(v_emb, t_emb.T) / temperature
    batch_size = logits.size(0)
    
    # Mask for negatives (all except diagonal)
    mask = torch.ones_like(logits).fill_diagonal_(0).bool()
    
    # Find hard negatives (highest similarity among negatives)
    negatives = logits.masked_fill(~mask, float('-inf'))
    num_hard = max(1, int(batch_size * hard_neg_ratio))
    
    # Standard symmetric loss
    labels = torch.arange(batch_size).to(v_emb.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    
    return (loss_v2t + loss_t2v) / 2

# =========================
# 1️⃣1️⃣ Optimizer Setup
# =========================
if config.finetune_lstm:
    # Different learning rates for LSTM and projection heads
    optimizer = torch.optim.AdamW([
        {'params': video_encoder.parameters(), 'lr': config.learning_rate * config.lstm_lr_multiplier},
        {'params': proj_video.parameters(), 'lr': config.learning_rate},
        {'params': proj_text.parameters(), 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
else:
    # Only train projection heads
    optimizer = torch.optim.AdamW(
        list(proj_video.parameters()) + list(proj_text.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs, eta_min=1e-6
)

# =========================
# 1️⃣2️⃣ Training and Validation Functions
# =========================
def train_epoch(epoch):
    if config.finetune_lstm:
        video_encoder.train()
    proj_video.train()
    proj_text.train()
    
    total_loss = 0
    total_v2t = 0
    total_t2v = 0
    
    for batch_idx, (video_seq, text, _) in enumerate(train_loader):
        video_seq = video_seq.to(config.device)
        
        # Video encoding
        if config.finetune_lstm:
            v_feat = video_encoder(video_seq)
        else:
            with torch.no_grad():
                v_feat = video_encoder(video_seq)
        
        # Text encoding
        t_feat = torch.tensor(
            text_encoder.encode(text, convert_to_numpy=True),
            dtype=torch.float32
        ).to(config.device)
        
        # Project and normalize
        v_emb = F.normalize(proj_video(v_feat), dim=1)
        t_emb = F.normalize(proj_text(t_feat), dim=1)
        
        # Compute loss
        if config.use_hard_negatives and len(video_seq) > 4:
            loss = hard_negative_loss(v_emb, t_emb, config.temperature)
            loss_v2t = loss_t2v = loss  # For logging
        else:
            loss, loss_v2t, loss_t2v = contrastive_loss(v_emb, t_emb, config.temperature)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(proj_video.parameters()) + list(proj_text.parameters()), 
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        total_v2t += loss_v2t.item()
        total_t2v += loss_t2v.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_v2t = total_v2t / len(train_loader)
    avg_t2v = total_t2v / len(train_loader)
    
    return avg_loss, avg_v2t, avg_t2v

def validate():
    video_encoder.eval()
    proj_video.eval()
    proj_text.eval()
    
    total_loss = 0
    total_v2t = 0
    total_t2v = 0
    
    with torch.no_grad():
        for video_seq, text, _ in val_loader:
            video_seq = video_seq.to(config.device)
            
            # Encode
            v_feat = video_encoder(video_seq)
            t_feat = torch.tensor(
                text_encoder.encode(text, convert_to_numpy=True),
                dtype=torch.float32
            ).to(config.device)
            
            # Project and normalize
            v_emb = F.normalize(proj_video(v_feat), dim=1)
            t_emb = F.normalize(proj_text(t_feat), dim=1)
            
            # Compute loss
            loss, loss_v2t, loss_t2v = contrastive_loss(v_emb, t_emb, config.temperature)
            
            total_loss += loss.item()
            total_v2t += loss_v2t.item()
            total_t2v += loss_t2v.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_v2t = total_v2t / len(val_loader)
    avg_t2v = total_t2v / len(val_loader)
    
    return avg_loss, avg_v2t, avg_t2v

# =========================
# 1️⃣3️⃣ Training Loop
# =========================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

best_val_loss = float('inf')
history = {
    'train_loss': [],
    'val_loss': [],
    'train_v2t': [],
    'train_t2v': [],
    'val_v2t': [],
    'val_t2v': []
}

for epoch in range(config.epochs):
    # Train
    train_loss, train_v2t, train_t2v = train_epoch(epoch)
    
    # Validate
    val_loss, val_v2t, val_t2v = validate()
    
    # Update scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_v2t'].append(train_v2t)
    history['train_t2v'].append(train_t2v)
    history['val_v2t'].append(val_v2t)
    history['val_t2v'].append(val_t2v)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{config.epochs}] LR: {current_lr:.6f}")
    print(f"  Train - Loss: {train_loss:.4f} | V2T: {train_v2t:.4f} | T2V: {train_t2v:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f} | V2T: {val_v2t:.4f} | T2V: {val_t2v:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        # Save all models
        torch.save(proj_video.state_dict(), config.model_dir / "proj_video_best.pth")
        torch.save(proj_text.state_dict(), config.model_dir / "proj_text_best.pth")
        
        if config.finetune_lstm:
            torch.save(video_encoder.state_dict(), config.model_dir / "video_encoder_best.pth")
        
        print(f"  ✅ Best model saved! (Val loss: {best_val_loss:.4f})")
    
    print()

# =========================
# 1️⃣4️⃣ Save Final Models
# =========================
print("="*80)
print("TRAINING COMPLETED!")
print("="*80)

# Save final models
torch.save(proj_video.state_dict(), config.model_dir / "proj_video_final.pth")
torch.save(proj_text.state_dict(), config.model_dir / "proj_text_final.pth")

if config.finetune_lstm:
    torch.save(video_encoder.state_dict(), config.model_dir / "video_encoder_final.pth")

print(f"\n✅ Final models saved to {config.model_dir}/")
print(f"   - proj_video_final.pth")
print(f"   - proj_text_final.pth")
if config.finetune_lstm:
    print(f"   - video_encoder_final.pth")

print(f"\n✅ Best models saved (Val loss: {best_val_loss:.4f}):")
print(f"   - proj_video_best.pth")
print(f"   - proj_text_best.pth")
if config.finetune_lstm:
    print(f"   - video_encoder_best.pth")

# Save training history
with open(config.model_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\n✅ Training history saved to {config.model_dir / 'training_history.json'}")

# =========================
# 1️⃣5️⃣ Quick Validation Check
# =========================
print("\n" + "="*80)
print("QUICK VALIDATION CHECK")
print("="*80)

# Load best models
proj_video.load_state_dict(torch.load(config.model_dir / "proj_video_best.pth"))
proj_text.load_state_dict(torch.load(config.model_dir / "proj_text_best.pth"))

if config.finetune_lstm:
    video_encoder.load_state_dict(torch.load(config.model_dir / "video_encoder_best.pth"))

video_encoder.eval()
proj_video.eval()
proj_text.eval()

# Test samples
test_texts = [
    "forklift lifting the cylindrical load upward",
    "forklift placing the load on the ground",
    "person walking with a dog",  # Anomaly
    "car driving on highway"  # Anomaly
]

with torch.no_grad():
    # Get a few video samples
    test_sequences = torch.tensor(sequences[:5], dtype=torch.float32).to(config.device)
    v_feat = video_encoder(test_sequences)
    v_emb = F.normalize(proj_video(v_feat), dim=1)
    
    print("\nSimilarity Matrix (5 video samples vs 4 texts):")
    print("-" * 80)
    
    for text in test_texts:
        t_feat = torch.tensor(
            text_encoder.encode([text], convert_to_numpy=True),
            dtype=torch.float32
        ).to(config.device)
        t_emb = F.normalize(proj_text(t_feat), dim=1)
        
        similarities = torch.matmul(v_emb, t_emb.T).squeeze()
        max_sim = similarities.max().item()
        mean_sim = similarities.mean().item()
        
        status = "✅ NORMAL" if max_sim > 0.3 else "⚠️ ANOMALY"
        print(f"\n{status} Text: '{text[:50]}...'")
        print(f"  Max similarity: {max_sim:.4f}")
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Similarities: {similarities.cpu().numpy()}")

print("\n" + "="*80)
print("Training script completed successfully! ✅")
print("="*80)