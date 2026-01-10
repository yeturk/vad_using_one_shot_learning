import os
import torch
import clip
from PIL import Image

# ============================================
# 1) CLIP MODEL
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

K_FRAMES = 8  # her segment için kullanılacak frame sayısı

# ============================================
# 2) IPAD DATASET YAPISI
# ============================================

# Örn:
# ../data/IPAD_dataset/R03/training/frames/
#     ├─ 01/
#     ├─ 02/
#     ├─ 03/
#     └─ 04/  ...
BASE_DIR = "../data/IPAD_dataset/R03/training/frames"

# Bu klasördeki tüm video klasörlerini (01, 02, 03, ...) topla
VIDEO_IDS = sorted([d for d in os.listdir(BASE_DIR) if d.isdigit()])
print("Bulunan video klasörleri:", VIDEO_IDS)


# ============================================
# 3) EPISODE_CONFIG YAPISI (ÖNEMLİ KISIM)
# ============================================
"""
Her sınıf için yapı şöyle:

"class_name": {
    "text": [...],  # o aksiyonu anlatan 1+ prompt
    "support": [
        {"video": "01", "start_frame": 0,   "end_frame": 60},
        {"video": "02", "start_frame": 0,   "end_frame": 60},
    ],
    "query": [
        {"video": "03", "start_frame": 0,   "end_frame": 60},
        {"video": "04", "start_frame": 0,   "end_frame": 60},
    ]
}

- "video": IPAD içindeki alt klasör adı (örn. "01", "02", ...)
- "start_frame" / "end_frame": bu aksiyonun o videodaki frame aralığı
  (0-index, python slice mantığı: [start, end) )
"""

EPISODE_CONFIG = {
    "approach_load": {
        "text": [
            "forklift approaching the cylindrical load",
            "forklift moving toward the load from a distance"
        ],
        "support": [
            {"video": "01", "start_frame": 0,   "end_frame": 60},
            {"video": "02", "start_frame": 0,   "end_frame": 60},
        ],
        "query": [
            {"video": "03", "start_frame": 0,   "end_frame": 60},
        ],
    },

    "lift_load": {
        "text": [
            "forklift lifting the cylindrical load upward",
            "forklift raising the load with its forks"
        ],
        "support": [
            {"video": "01", "start_frame": 60,  "end_frame": 120},
            {"video": "02", "start_frame": 60,  "end_frame": 120},
        ],
        "query": [
            {"video": "03", "start_frame": 60,  "end_frame": 120},
        ],
    },

    "move_with_load": {
        "text": [
            "forklift moving backward while carrying the raised load",
            "forklift transporting the elevated load"
        ],
        "support": [
            {"video": "01", "start_frame": 120, "end_frame": 180},
            {"video": "02", "start_frame": 120, "end_frame": 180},
        ],
        "query": [
            {"video": "03", "start_frame": 120, "end_frame": 180},
        ],
    },

    "lower_and_leave": {
        "text": [
            "forklift lowering the load to the ground and leaving",
            "forklift placing the load down and moving away"
        ],
        "support": [
            {"video": "01", "start_frame": 180, "end_frame": 240},
            {"video": "02", "start_frame": 180, "end_frame": 240},
        ],
        "query": [
            {"video": "03", "start_frame": 180, "end_frame": 240},
        ],
    },
}

CLASS_NAMES = list(EPISODE_CONFIG.keys())


# ============================================
# 4) HELPER: BİR VİDEODAKİ FRAME LİSTESİ
# ============================================
def list_video_frames(video_id: str):
    """Bir video klasöründeki frame dosyalarını sıralı döndürür."""
    vdir = os.path.join(BASE_DIR, video_id)
    frames = sorted(
        f for f in os.listdir(vdir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return vdir, frames


def sample_frames_from_range(video_id: str, start_frame: int, end_frame: int, k: int = K_FRAMES):
    """
    Belirli bir video ve frame aralığından (start_frame, end_frame)
    k adet frame örnekler (eşit aralıklı).
    """
    vdir, frames = list_video_frames(video_id)
    n = len(frames)
    if n == 0:
        return vdir, []

    # Aralığı dataset sınırları ile kesiştir
    start = max(0, start_frame)
    end = min(end_frame, n)
    if start >= end:
        return vdir, []

    sub_frames = frames[start:end]
    if len(sub_frames) <= k:
        return vdir, sub_frames

    # k'tan fazlaysa eşit aralıklı örnekle
    idx = torch.linspace(0, len(sub_frames) - 1, k).long()
    sampled = [sub_frames[i] for i in idx]
    return vdir, sampled


# ============================================
# 5) CLIP EMBEDDING FONKSİYONLARI
# ============================================
def embed_frames(video_dir: str, frame_list):
    """Verilen frame listesinin CLIP image embedding ortalamasını döndürür."""
    if not frame_list:
        return None

    feats = []
    for fname in frame_list:
        img_path = os.path.join(video_dir, fname)
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
        feats.append(feat)

    return torch.mean(torch.cat(feats, dim=0), dim=0, keepdim=True)


def embed_texts(text_list):
    """Bir sınıf için 1+ prompt'u encode edip ortalamasını döndürür."""
    tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        tfeat = model.encode_text(tokens)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
    return torch.mean(tfeat, dim=0, keepdim=True)


# ============================================
# 6) SUPPORT PROTOTYPE OLUŞTURMA
# ============================================
print("\n=== SUPPORT PROTOTYPES OLUŞTURULUYOR ===")
class_prototypes = {}
class_text_embeddings = {}

for cname in CLASS_NAMES:
    cfg = EPISODE_CONFIG[cname]

    # Text embedding (raporda kullanmak istersen)
    class_text_embeddings[cname] = embed_texts(cfg["text"])

    # Support segmentlerinden image prototipi
    support_embs = []
    for seg in cfg["support"]:
        v = seg["video"]
        s = seg["start_frame"]
        e = seg["end_frame"]

        vdir, flist = sample_frames_from_range(v, s, e, k=K_FRAMES)
        emb = embed_frames(vdir, flist)
        if emb is not None:
            support_embs.append(emb)

    if not support_embs:
        # Hiç yoksa sıfır vektör (güvenlik)
        dim = class_text_embeddings[cname].shape[-1]
        class_prototypes[cname] = torch.zeros((1, dim), device=device)
        print(f"[UYARI] Sınıf {cname} için support embedding bulunamadı!")
    else:
        proto = torch.mean(torch.cat(support_embs, dim=0), dim=0, keepdim=True)
        class_prototypes[cname] = proto
        print(f"Sınıf {cname}: {len(support_embs)} support segment → prototype oluşturuldu.")


# ============================================
# 7) QUERY ÜZERİNDE TEST
# ============================================
print("\n=== QUERY TEST BAŞLIYOR ===")
correct = 0
total = 0

for cname in CLASS_NAMES:
    cfg = EPISODE_CONFIG[cname]

    for seg in cfg["query"]:
        v = seg["video"]
        s = seg["start_frame"]
        e = seg["end_frame"]

        vdir, flist = sample_frames_from_range(v, s, e, k=K_FRAMES)
        q_emb = embed_frames(vdir, flist)
        if q_emb is None:
            continue

        # Tüm sınıf prototipleriyle cosine similarity
        sims = []
        for cname2 in CLASS_NAMES:
            proto = class_prototypes[cname2]
            sim = (q_emb @ proto.T).item()
            sims.append((cname2, sim))

        pred_class, pred_sim = max(sims, key=lambda x: x[1])

        total += 1
        if pred_class == cname:
            correct += 1

        print(f"Gerçek: {cname:15s} | Tahmin: {pred_class:15s} | Sim: {pred_sim:.3f}")

if total > 0:
    acc = correct / total * 100
    print("\n==============================")
    print(f" Few-Shot Prototype Accuracy: {acc:.2f}%  ({correct}/{total})")
    print("==============================")
else:
    print("\nQuery örneği bulunamadı. EPISODE_CONFIG'i kontrol et.")
