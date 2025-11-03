import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

print("\n ************    0. evaluate_lstm_ipad.py is executed by yet :)    ************")

# ======================================================
# 1️⃣ Load anomaly scores & ground truth
# ======================================================
scores = np.load("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01.npy")
labels = np.load("/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/001.npy")

# İkisini hizalamak için sequence offset düzeltmesi
# (her sequence 10 frame olduğundan son 10 frame labeli düşer)
min_len = min(len(scores), len(labels))
scores = scores[:min_len]
labels = labels[:min_len]

print(f"✅ Scores shape: {scores.shape}, Labels shape: {labels.shape}")

# ======================================================
# 2️⃣ Normalize Scores
# ======================================================
scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# ======================================================
# 3️⃣ Determine threshold
# ======================================================

# option-1
# threshold = np.percentile(scores_norm, 90)
# print(f"⚙️  Threshold (90 percentile): {threshold:.4f}")

# option-2
# ======================================
#  Compute Threshold using Z-score rule
# ======================================
# mean = np.mean(scores_norm)
# std = np.std(scores_norm)
# threshold = mean + 1.5 * std  # or try 1.2 – 1.8 range for tuning

# print(f"Mean: {mean:.3f}, Std: {std:.3f}, Threshold (Z-Score): {threshold:.3f}")

# option-3
from sklearn.metrics import precision_recall_curve, f1_score

precisions, recalls, thresholds = precision_recall_curve(labels, scores_norm)
f1s = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1s)
best_thresh = thresholds[best_idx]
print(f"🔍 Best threshold for max F1: {best_thresh:.3f}")

threshold = best_thresh
preds = (scores_norm > threshold).astype(int)

# ======================================================
# 4️⃣ Metrics
# ======================================================
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)
auc = roc_auc_score(labels, scores_norm)

print(f"""
📊 Evaluation Metrics:
-----------------------
✅ Accuracy : {acc:.3f}
✅ Precision: {prec:.3f}
✅ Recall   : {rec:.3f}
✅ F1 Score : {f1:.3f}
✅ ROC-AUC  : {auc:.3f}
""")

# ======================================================
# 5️⃣ Visualization
# ======================================================
plt.figure(figsize=(12,5))
plt.plot(scores_norm, label="Anomaly Score (normalized)")
plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold={threshold:.3f}")
plt.fill_between(np.arange(len(labels)), 0, 1, where=labels>0, color='red', alpha=0.2, label="Ground Truth Anomaly")
plt.xlabel("Sequence Index (time)")
plt.ylabel("Normalized Error")
plt.title("Ground Truth vs Predicted Anomaly Scores")
plt.legend()
plt.grid(True)
plt.show()