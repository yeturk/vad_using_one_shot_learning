import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

print("\n ************    0. extract_test_features_ipad.py is executed by yet :)    ************")

# ======================================================
# 1. Device Checking
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ======================================================
# 2. Load Pretrained CNN Model (EfficientNet-B0)
# ======================================================
print("\n ************    1. Loading EfficientNet-B0    ************")

# base_model = models.efficientnet_b0(pretrained=True) → DEPRECATED WARNING

from torchvision.models import EfficientNet_B0_Weights
base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Classification katmanını kaldır (sadece feature extraction)
model = nn.Sequential(*list(base_model.children())[:-1])
model.to(device)
model.eval()
print("✅ Model loaded successfully.\n")

# ======================================================
# 3. Define Preprocessing
# ======================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================================================
# 4. Define Directories (TEST PATHS)
# ======================================================
# the first test  → R01/testing/frames/01
# frames_dir = "dataset/IPAD_dataset/R01/testing/frames/01"
# output_path = "dataset/IPAD_dataset/R01/testing/features"
# the second test → R01/testing/frames/06
frames_dir = "dataset/IPAD_dataset/R01/testing/frames/06"
output_path = "dataset/IPAD_dataset/R01/testing/features"
os.makedirs(output_path, exist_ok=True)

print(f"📂 Reading test frames from: {frames_dir}")

# ======================================================
# 5. Feature Extraction
# ======================================================
features = []
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

for frame_name in tqdm(frame_files, desc="Extracting test features"):
    frame_path = os.path.join(frames_dir, frame_name)
    image = Image.open(frame_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(input_tensor).squeeze().cpu().numpy()
        features.append(feature)

features = np.array(features)
print(f"✅ Test features extracted: {features.shape}")

# ======================================================
# 6. Save Output
# ======================================================
# output_file = os.path.join(output_path, "01.npy")
output_file = os.path.join(output_path, "06.npy")
np.save(output_file, features)
print(f"💾 Saved test features to: {output_file}")

print("\n 🎯 Test feature extraction completed successfully!")