import numpy as np
import matplotlib.pyplot as plt

# test dosyaları
scores = np.load("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01_06.npy")
labels = np.load("/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/006.npy")

print("✅ Scores shape:", scores.shape)
print("✅ Labels shape:", labels.shape)

# sequence length (10 frame varsayımı)
seq_len  = 10
n_frames = len(labels)
n_seq    = len(scores)

expected_seq = n_frames - seq_len
print("Expected sequence count:", expected_seq)
print("Sequence-label farkı:", n_frames - n_seq)

print("=" * 70 + "\n")
# =================================================================================

# scores_normal  = np.load("data/results/anomaly_scores_R01_09.npy")
# scores_anomaly = np.load("data/results/anomaly_scores_R01_06.npy")

# plt.figure(figsize=(8,5))
# plt.hist(scores_normal,  bins=30, alpha=0.6, label="Normal (R01_09)")
# plt.hist(scores_anomaly, bins=30, alpha=0.6, label="Anomaly (R01_06)")
# plt.xlabel("Reconstruction Error (MSE)")
# plt.ylabel("Frequency")
# plt.title("Score Distributions: Normal vs Anomaly")
# plt.legend()
# plt.show()
# =================================================================================


# def frame_to_sequence_labels(frame_labels, seq_len=10):
#     """Convert frame-level labels to sequence-level labels."""
#     seq_labels = []
#     for i in range(len(frame_labels) - seq_len):
#         window = frame_labels[i : i + seq_len]
#         seq_labels.append(1 if np.any(window == 1) else 0)
#     return np.array(seq_labels)


# # örnek:
# labels = np.load("/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/006.npy")
# seq_labels = frame_to_sequence_labels(labels, seq_len=10)
# print(f"Frame labels: {len(labels)}, Sequence labels: {len(seq_labels)}")

print("=" * 70 + "\n")
# =================================================================================

scores_06 = np.load("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01_06.npy")
scores_09 = np.load("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01_09.npy")

plt.figure(figsize=(8,5))
plt.hist(scores_06, bins=30, alpha=0.6, label="Anomaly (06)")
plt.hist(scores_09, bins=30, alpha=0.6, label="Normal (09)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Distribution Comparison: Normal vs Anomaly")
plt.legend()
plt.show()
