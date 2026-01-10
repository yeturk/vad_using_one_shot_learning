import os
import torch
import torch.nn as nn
from PIL import Image
import clip

# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# CLIP MODEL
# ============================================================
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Tüm modeli float32'ye çek (daha stabil grad için)
clip_model.float()

# Tüm residual block'lardaki attn_mask'i None yap
for block in clip_model.transformer.resblocks:
    block.attn_mask = None

# Maks token sayısı (77)
MAX_INPUT_TOKENS = clip_model.positional_embedding.shape[0]

# ============================================================
# CoOP: Learnable Context Parameters
# ============================================================
CONTEXT_LEN = 4
EMB_DIM = clip_model.text_projection.shape[1]


class LearnablePrompt(nn.Module):
    def __init__(self, classnames, ctx_len=CONTEXT_LEN, embed_dim=EMB_DIM):
        """
        classnames: list of text prompts, örn:
            ["worker approaches the load", "worker lifts the load", ...]
        """
        super().__init__()
        self.classnames = classnames
        self.ctx_len = ctx_len

        # Sınıf adı için izin verilen maksimum token uzunluğu: 77 - ctx_len
        self.max_allowed_name_len = MAX_INPUT_TOKENS - self.ctx_len
        self.max_name_len = self.max_allowed_name_len

        # ctx'i float32 olarak başlat
        ctx_initial = torch.randn(ctx_len, embed_dim, dtype=torch.float32)
        self.ctx = nn.Parameter(ctx_initial)

        # Sınıf isimlerinin embedding'leri (token embedding sonrası)
        self.name_embeds = {}

        with torch.no_grad():
            for cname in classnames:
                toks = clip.tokenize(cname).to(device)   # [1, T]
                # token embedding: [1, T, d] -> [T, d]
                emb = clip_model.token_embedding(toks).squeeze(0)
                emb = emb.to(torch.float32)

                # Sınıf adını max_allowed_name_len ile sınırla
                if emb.shape[0] > self.max_allowed_name_len:
                    emb = emb[:self.max_allowed_name_len]

                self.name_embeds[cname] = emb

    def pad_name(self, emb):
        """
        Class-name token embedding'lerini max_name_len'e pad ederek eşitle.
        """
        t, d = emb.shape
        if t == self.max_name_len:
            return emb

        pad_len = self.max_name_len - t
        pad = torch.zeros((pad_len, d), device=device, dtype=emb.dtype)
        emb = torch.cat([emb, pad], dim=0)
        return emb

    def forward(self):
        """
        Her sınıf için: [1, 77, d] shape'inde prompt embedding döndür.
        """
        prompts = {}
        ctx = self.ctx  # [ctx_len, d] (float32)

        for cname in self.classnames:
            name_emb = self.name_embeds[cname]   # [t <= max_name_len, d]
            name_emb = self.pad_name(name_emb)   # [max_name_len, d]

            # Tam uzunluk: ctx_len + max_name_len = 77
            full = torch.cat(
                [ctx, name_emb], dim=0
            ).unsqueeze(0)  # [1, 77, d]

            prompts[cname] = full.to(device)

        return prompts


# ============================================================
# HELPERS: FRAME LOADING
# ============================================================
def list_frames(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def load_frames_segment(folder, start_frame, end_frame, k=8):
    """
    Belirli frame aralığından K frame seç.
    Linearly spaced örnekleme kullanılıyor.
    """
    frames = list_frames(folder)
    frames = frames[start_frame:end_frame]

    if len(frames) == 0:
        return []

    if len(frames) <= k:
        selected = frames
    else:
        idx = torch.linspace(0, len(frames) - 1, k).long()
        selected = [frames[i] for i in idx]

    imgs = [
        Image.open(os.path.join(folder, f)).convert("RGB")
        for f in selected
    ]
    return imgs


@torch.no_grad()
def encode_image(imgs):
    """
    Bir segmentteki K adet frame'den CLIP image embedding hesapla.
    Sonuç: [1, d]
    """
    feats = []
    for img in imgs:
        t = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W]
        t = t.float()
        feat = clip_model.encode_image(t)           # [1, d]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat)

    feats = torch.cat(feats, dim=0)  # [K, d]
    return feats.mean(dim=0, keepdim=True)  # [1, d]


# ============================================================
# TEXT ENCODING (CoOP Prompt)
# ============================================================
def encode_text_from_prompt(prompt_emb):
    """
    prompt_emb shape: [1, T, d] (float32)
    CLIP text encoder mantığına göre çıktı: [1, d]
    """
    # Positional embedding'i al ve dtype/device ayarla
    pos = clip_model.positional_embedding.to(device=device, dtype=prompt_emb.dtype)
    # [T, d] -> [1, T, d]
    pos = pos.unsqueeze(0)

    x = prompt_emb + pos  # [1, T, d]

    # CLIP transformer float32 çalışıyor
    x = clip_model.transformer(x)

    # CLIP text encoder son token'ı alıyor
    x = x[:, -1, :]  # [1, d]

    # Projection
    x = x @ clip_model.text_projection  # [1, d]

    # Normalize (yerinde olmayan, clone yok)
    x = x / x.norm(dim=-1, keepdim=True)

    return x  # [1, d]


# ============================================================
# FEW-SHOT EPISODE CONFIG (Gerçek IPAD yapısına uygun)
# ============================================================
BASE_DIR = "../data/IPAD_dataset/R03/training/frames"

EPISODE_CONFIG = {
    # Toplam 22 video (01'den 22'ye kadar) kullanıldı.
    
    "worker approaches the load": {
        "support": [
            # 15 Support Video Segmenti (01'den 15'e kadar videoların 0-60. frameleri)
            {"video": str(i).zfill(2), "start_frame": 0, "end_frame": 60} for i in range(1, 16)
        ],
        "query": [
            # 7 Query Video Segmenti (16'dan 22'ye kadar videoların 0-60. frameleri)
            {"video": str(i).zfill(2), "start_frame": 0, "end_frame": 60} for i in range(16, 23)
        ]
    },

    "worker lifts the load": {
        "support": [
            # 15 Support Video Segmenti (01'den 15'e kadar videoların 60-120. frameleri)
            {"video": str(i).zfill(2), "start_frame": 60, "end_frame": 120} for i in range(1, 16)
        ],
        "query": [
            # 7 Query Video Segmenti (16'dan 22'ye kadar videoların 60-120. frameleri)
            {"video": str(i).zfill(2), "start_frame": 60, "end_frame": 120} for i in range(16, 23)
        ]
    },

    "worker moves with the load": {
        "support": [
            # 15 Support Video Segmenti (01'den 15'e kadar videoların 120-180. frameleri)
            {"video": str(i).zfill(2), "start_frame": 120, "end_frame": 180} for i in range(1, 16)
        ],
        "query": [
            # 7 Query Video Segmenti (16'dan 22'ye kadar videoların 120-180. frameleri)
            {"video": str(i).zfill(2), "start_frame": 120, "end_frame": 180} for i in range(16, 23)
        ]
    },

    "worker lowers and leaves the load": {
        "support": [
            # 15 Support Video Segmenti (01'den 15'e kadar videoların 180-240. frameleri)
            {"video": str(i).zfill(2), "start_frame": 180, "end_frame": 240} for i in range(1, 16)
        ],
        "query": [
            # 7 Query Video Segmenti (16'dan 22'ye kadar videoların 180-240. frameleri)
            {"video": str(i).zfill(2), "start_frame": 180, "end_frame": 240} for i in range(16, 23)
        ]
    },
}

CLASS_NAMES = list(EPISODE_CONFIG.keys())

prompt_model = LearnablePrompt(CLASS_NAMES).to(device)

# ============================================================
# SUPPORT EMBEDDING'LERİNİ ÖNCEDEN HESAPLA (CACHE)
# ============================================================
print("\n=== PRECOMPUTING SUPPORT IMAGE EMBEDDINGS ===")
support_cache = {}

with torch.no_grad():
    for cname in CLASS_NAMES:
        sup_embeds = []
        for seg in EPISODE_CONFIG[cname]["support"]:
            folder = os.path.join(BASE_DIR, seg["video"])
            imgs = load_frames_segment(
                folder,
                seg["start_frame"],
                seg["end_frame"],
                k=8
            )
            if len(imgs) == 0:
                continue
            emb = encode_image(imgs)  # [1, d]
            sup_embeds.append(emb)

        if len(sup_embeds) == 0:
            # Güvenlik için, eğer hiç frame yoksa random vektör koy
            support_cache[cname] = torch.randn(1, EMB_DIM, device=device)
        else:
            image_feat = torch.cat(sup_embeds, dim=0).mean(dim=0, keepdim=True)
            support_cache[cname] = image_feat  # [1, d]

# ============================================================
# TRAINING LOOP (CoOP Prompt Optimization)
# ============================================================
optimizer = torch.optim.Adam([prompt_model.ctx], lr=1e-3)
epochs = 50

print("\n=== TRAINING CoOP PROMPTS ===")

for epoch in range(epochs):
    total_loss = 0.0
    prompt_dict = prompt_model()  # her sınıf için [1,77,d]

    for cname in CLASS_NAMES:
        # Support image embedding (önceden hesaplanmış)
        image_feat = support_cache[cname]  # [1, d]

        # Tüm sınıf text embeddingleri
        text_embs = []
        for cname2 in CLASS_NAMES:
            t_emb = encode_text_from_prompt(prompt_dict[cname2])  # [1, d]
            text_embs.append(t_emb)
        text_embs = torch.cat(text_embs, dim=0)  # [C, d]

        # Similarity
        sim = image_feat @ text_embs.T  # [1, C]

        # CLIP logit_scale (grad gerekmediği için detach)
        logit_scale = clip_model.logit_scale.exp().detach()
        sim = sim * logit_scale

        label = torch.tensor(
            [CLASS_NAMES.index(cname)],
            device=device,
            dtype=torch.long
        )

        loss = nn.CrossEntropyLoss()(sim, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# ============================================================
# TEST (QUERY FOLDERS)
# ============================================================
print("\n=== TESTING ===")
correct = 0
total = 0

prompt_dict = prompt_model()  # eğitilmiş ctx ile yeniden hesapla

for cname in CLASS_NAMES:
    for seg in EPISODE_CONFIG[cname]["query"]:
        folder = os.path.join(BASE_DIR, seg["video"])
        imgs = load_frames_segment(
            folder,
            seg["start_frame"],
            seg["end_frame"],
            k=8
        )
        if len(imgs) == 0:
            continue

        q_emb = encode_image(imgs)  # [1, d]

        text_embs = []
        for cname2 in CLASS_NAMES:
            t_emb = encode_text_from_prompt(prompt_dict[cname2])
            text_embs.append(t_emb)
        text_embs = torch.cat(text_embs, dim=0)  # [C, d]

        sim = (q_emb @ text_embs.T).squeeze(0)  # [C]
        pred_idx = sim.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]

        total += 1
        if pred_class == cname:
            correct += 1

        print(f"Gerçek: {cname:35s} | Tahmin: {pred_class:35s}")

accuracy = correct / total * 100 if total > 0 else 0.0
print("\n===============================")
print(f"CoOP Few-Shot Accuracy: {accuracy:.2f}% ({correct}/{total})")
print("===============================")
