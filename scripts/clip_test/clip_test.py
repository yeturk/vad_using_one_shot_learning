import os
import torch
import clip
from PIL import Image

# ===========================
# 1) CLIP MODEL YÜKLE
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)


# ===========================
# 2) GÜNCELLENMİŞ CLIP-OPTIMIZE TEXT LİSTESİ
# ===========================
text_list = [
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



num_seconds = len(text_list)
fps = 30

# Tokenize text once
text_tokens = clip.tokenize(text_list).to(device)


# ===========================
# 3) FRAME DİZİNİ
# ===========================
frame_dir = "../data/IPAD_dataset/R03/training/frames/01"
frames = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg",".jpeg",".png")))
total_frames = len(frames)

print(f"Toplam frame: {total_frames}")
print(f"Toplam saniye: {num_seconds}")


# ===========================
# 4) TEXT EMBEDDING ÖN HESAPLAMA
# ===========================
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# ===========================
# 5) HELPER FONKSİYONLAR
# ===========================
def get_second_frames(sec):
    start = sec * fps
    end = min((sec + 1) * fps, total_frames)
    return frames[start:end]


def encode_frame(path):
    img = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat


# ===========================
# 6) SIMILARITY + ACCURACY HESABI
# ===========================
correct = 0

for sec in range(num_seconds):
    frame_list = get_second_frames(sec)
    if not frame_list:
        continue

    feats = [encode_frame(os.path.join(frame_dir, f)) for f in frame_list]
    video_feat = torch.mean(torch.cat(feats, dim=0), dim=0, keepdim=True)

    sim = (video_feat @ text_features.T).squeeze(0)
    predicted_sec = torch.argmax(sim).item()

    print(f"Saniye {sec:02d} -> Tahmin: {predicted_sec:02d} | Doğru text skoru: {sim[sec]:.3f}")

    if predicted_sec == sec:
        correct += 1

accuracy = correct / num_seconds * 100
print("\n===========================")
print(f"DOĞRULUK (Accuracy): {accuracy:.2f}%")
print("===========================")
