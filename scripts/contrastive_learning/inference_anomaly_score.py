import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import json
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

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
    
    # Inference
    window_size = 3  # seconds
    threshold_percentile = 10  # Use 10th percentile of similarities as threshold
    
    # Paths
    model_dir = Path("../models")
    output_dir = Path("../outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model selection
    use_best_model = True  # True: use best, False: use final

config = Config()
print(f"Device: {config.device}")
print(f"Using {'best' if config.use_best_model else 'final'} model")

# =========================
# 2️⃣ LSTM Encoder
# =========================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return h[-1]

# =========================
# 3️⃣ Load Models
# =========================
print("\n" + "="*80)
print("LOADING MODELS")
print("="*80)

# Determine which models to load
model_suffix = "best" if config.use_best_model else "final"

# Video encoder
video_encoder = LSTMEncoder(config.input_dim, config.hidden_dim).to(config.device)

try:
    # Try to load fine-tuned encoder first
    encoder_path = config.model_dir / f"video_encoder_{model_suffix}.pth"
    if encoder_path.exists():
        video_encoder.load_state_dict(torch.load(encoder_path, map_location=config.device))
        print(f"✅ Loaded fine-tuned video encoder: {encoder_path}")
    else:
        # Fallback to original LSTM
        video_encoder.load_state_dict(
            torch.load("models/lstm_ipad.pth", map_location=config.device),
            strict=False
        )
        print("✅ Loaded original pre-trained LSTM")
except Exception as e:
    print(f"⚠️ Warning: Could not load video encoder - {e}")
    print("Using randomly initialized encoder")

video_encoder.eval()

# Projection heads
proj_video = nn.Linear(config.hidden_dim, config.projection_dim).to(config.device)
proj_video.load_state_dict(
    torch.load(config.model_dir / f"proj_video_{model_suffix}.pth", map_location=config.device)
)
proj_video.eval()
print(f"✅ Loaded video projection head: proj_video_{model_suffix}.pth")

proj_text = nn.Linear(config.text_dim, config.projection_dim).to(config.device)
proj_text.load_state_dict(
    torch.load(config.model_dir / f"proj_text_{model_suffix}.pth", map_location=config.device)
)
proj_text.eval()
print(f"✅ Loaded text projection head: proj_text_{model_suffix}.pth")

# Text encoder
text_encoder = SentenceTransformer(config.text_encoder_name, device=config.device)
print(f"✅ Loaded text encoder: {config.text_encoder_name}")

# =========================
# 4️⃣ Load Video Features
# =========================
print("\n" + "="*80)
print("LOADING VIDEO DATA")
print("="*80)

features = np.load(config.data_path)
print(f"Loaded features: {features.shape}")

sequence_length = config.sequence_length
sequences = []
frame_indices = []

for i in range(len(features) - sequence_length):
    sequences.append(features[i:i+sequence_length])
    frame_indices.append(i)

sequences = torch.tensor(np.array(sequences), dtype=torch.float32).to(config.device)
frame_indices = np.array(frame_indices)
print(f"✅ Created {len(sequences)} video sequences")

# =========================
# 5️⃣ Precompute Video Embeddings
# =========================
print("\n" + "="*80)
print("COMPUTING VIDEO EMBEDDINGS")
print("="*80)

video_embeddings = []

with torch.no_grad():
    batch_size = 64
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        v_feat = video_encoder(batch)
        v_emb = F.normalize(proj_video(v_feat), dim=1)
        video_embeddings.append(v_emb)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i+len(batch)}/{len(sequences)} sequences...")

video_embeddings = torch.cat(video_embeddings, dim=0)
print(f"✅ Video embeddings computed: {video_embeddings.shape}")

# =========================
# 6️⃣ Compute Baseline Statistics
# =========================
print("\n" + "="*80)
print("COMPUTING BASELINE STATISTICS")
print("="*80)

# Sample normal texts to establish baseline WITH temporal alignment
# More samples for better statistics
normal_text_second_pairs = [
    ("forklift approaching the cylindrical load on the left", 0),
    ("forklift aligned directly in front of the load", 1),
    ("forklift lowering its forks toward the load", 2),
    ("forklift sliding forks under the load", 3),
    ("forklift lifting the cylindrical load upward", 4),
    ("forklift holding the raised load", 5),
    ("forklift moving backward with the lifted load", 6),
    ("forklift transporting the elevated load backward", 7),
    ("forklift turning slightly while carrying the load", 8),
    ("forklift holding the load at maximum height", 9),
    ("forklift keeping the fully raised load stable", 10),
    ("forklift holding the elevated load near the top of the frame", 11),
    ("forklift adjusting position while keeping the load raised", 12),
    ("forklift holding the suspended load still", 13),
    ("forklift beginning to lower the load", 14),
    ("forklift lowering the load halfway", 15),
    ("forklift lowering the load near the ground", 16),
    ("forklift placing the load on the ground", 17),
    ("forklift retracting its forks", 18),
    ("forklift moving backward away from the load", 19),
    ("forklift reversing farther from the load", 20),
    ("forklift leaving the frame after placing the load", 21)
]

baseline_similarities = []

with torch.no_grad():
    for text, target_second in normal_text_second_pairs:
        t_feat = torch.tensor(
            text_encoder.encode([text], convert_to_numpy=True),
            dtype=torch.float32
        ).to(config.device)
        t_emb = F.normalize(proj_text(t_feat), dim=1)
        
        # Only use relevant temporal window for baseline
        target_frame = int(target_second * config.fps)
        window_frames = config.window_size * config.fps
        start_frame = max(0, target_frame - window_frames)
        end_frame = min(len(video_embeddings), target_frame + window_frames)
        
        # Compute similarity only with relevant frames
        relevant_video_embs = video_embeddings[start_frame:end_frame]
        similarities = torch.matmul(relevant_video_embs, t_emb.T).squeeze()
        
        # Use max similarity from the window (best match)
        max_sim = similarities.max().item()
        baseline_similarities.append(max_sim)

baseline_similarities = np.array(baseline_similarities)
baseline_mean = baseline_similarities.mean()
baseline_std = baseline_similarities.std()
baseline_threshold = np.percentile(baseline_similarities, config.threshold_percentile)

print(f"Baseline statistics from {len(normal_text_second_pairs)} normal text-second pairs:")
print(f"  Mean similarity: {baseline_mean:.4f}")
print(f"  Std similarity: {baseline_std:.4f}")
print(f"  {config.threshold_percentile}th percentile: {baseline_threshold:.4f}")
print(f"  Min: {baseline_similarities.min():.4f}")
print(f"  Max: {baseline_similarities.max():.4f}")
print(f"\n  Individual baseline similarities:")
for (text, second), sim in zip(normal_text_second_pairs, baseline_similarities):
    print(f"    {second:2d}s: {sim:.4f} | '{text[:50]}...'")


# =========================
# 7️⃣ Enhanced Anomaly Detection Function
# =========================
def detect_anomaly(
    text, 
    target_second, 
    fps=30, 
    threshold=None,
    window_size=3,
    method='statistical'
):
    """
    Enhanced anomaly detection with multiple scoring methods.
    
    Args:
        text (str): Text to test
        target_second (int/float): Target second in video
        fps (int): Video FPS
        threshold (float): Manual threshold (if None, uses baseline)
        window_size (int): Window size in seconds
        method (str): 'statistical' (z-score) or 'threshold' (simple)
        
    Returns:
        dict: Comprehensive anomaly analysis
    """
    # Use baseline threshold if not provided
    if threshold is None:
        threshold = baseline_threshold
    
    # Text embedding
    with torch.no_grad():
        t_feat = torch.tensor(
            text_encoder.encode([text], convert_to_numpy=True),
            dtype=torch.float32
        ).to(config.device)
        t_emb = F.normalize(proj_text(t_feat), dim=1)
    
    # Target frame range
    target_frame = int(target_second * fps)
    start_frame = max(0, target_frame - window_size * fps)
    end_frame = min(len(video_embeddings), target_frame + window_size * fps)
    
    # Compute similarities
    relevant_video_embs = video_embeddings[start_frame:end_frame]
    similarities = torch.matmul(t_emb, relevant_video_embs.T).squeeze()
    
    # Convert to numpy
    similarities_np = similarities.cpu().numpy()
    
    # Statistics
    max_similarity = similarities_np.max()
    mean_similarity = similarities_np.mean()
    std_similarity = similarities_np.std()
    median_similarity = np.median(similarities_np)
    
    # Z-score (statistical method)
    # Add minimum std to prevent extreme z-scores from very low variance
    effective_std = max(baseline_std, 0.05)  # Minimum 0.05 std
    z_score = (max_similarity - baseline_mean) / (effective_std + 1e-8)
    z_score_threshold = -1.5  # Below -1.5 std is anomalous
    
    # Anomaly detection
    if method == 'statistical':
        is_anomaly = z_score < z_score_threshold
        anomaly_score = -z_score  # Higher = more anomalous
        confidence = abs(z_score)
    else:  # threshold method
        is_anomaly = max_similarity < threshold
        anomaly_score = 1.0 - max_similarity
        confidence = abs(max_similarity - threshold)
    
    # Best match info
    best_match_idx = similarities_np.argmax()
    best_match_frame = start_frame + best_match_idx
    best_match_second = best_match_frame / fps
    
    # Percentile in baseline distribution
    percentile = (baseline_similarities < max_similarity).mean() * 100
    
    return {
        "text": text,
        "target_second": target_second,
        "window": f"{start_frame//fps}-{end_frame//fps}s",
        
        # Detection results
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(anomaly_score),
        "confidence": float(confidence),
        
        # Similarity statistics
        "max_similarity": float(max_similarity),
        "mean_similarity": float(mean_similarity),
        "median_similarity": float(median_similarity),
        "std_similarity": float(std_similarity),
        
        # Z-score analysis
        "z_score": float(z_score),
        "z_score_interpretation": (
            "Very anomalous" if z_score < -2 else
            "Anomalous" if z_score < -1.5 else
            "Borderline" if z_score < -1 else
            "Normal"
        ),
        
        # Baseline comparison
        "percentile_in_baseline": float(percentile),
        "distance_from_baseline_mean": float(max_similarity - baseline_mean),
        
        # Best match
        "best_match_second": float(best_match_second),
        "best_match_frame": int(best_match_frame),
        
        # Method info
        "detection_method": method,
        "threshold_used": float(threshold)
    }

# =========================
# 8️⃣ Batch Anomaly Detection
# =========================
def batch_detect_anomalies(
    text_second_pairs, 
    threshold=None,
    window_size=3,
    method='statistical',
    verbose=True
):
    """
    Batch anomaly detection with detailed reporting.
    
    Args:
        text_second_pairs (list): [(text, second), ...] pairs
        threshold (float): Detection threshold
        window_size (int): Window size in seconds
        method (str): Detection method
        verbose (bool): Print detailed results
        
    Returns:
        list: Results for each test
    """
    results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 BATCH ANOMALY DETECTION")
        print(f"{'='*80}")
        print(f"Method: {method}")
        print(f"Window size: {window_size}s")
        print(f"Threshold: {threshold if threshold else 'auto (baseline)'}")
        print(f"Total tests: {len(text_second_pairs)}")
        print(f"{'='*80}\n")
    
    for idx, (text, second) in enumerate(text_second_pairs, 1):
        result = detect_anomaly(
            text, second, 
            threshold=threshold,
            window_size=window_size,
            method=method
        )
        results.append(result)
        
        if verbose:
            # Status emoji
            if result["is_anomaly"]:
                if result["z_score"] < -2:
                    status = "🚨 STRONG ANOMALY"
                else:
                    status = "⚠️ ANOMALY"
            else:
                status = "✅ NORMAL"
            
            print(f"Test {idx}/{len(text_second_pairs)}")
            print(f"  Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
            print(f"  Target: {second}s | Window: {result['window']}")
            print(f"  Status: {status}")
            print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
            print(f"  Max Similarity: {result['max_similarity']:.4f}")
            print(f"  Z-score: {result['z_score']:.4f} ({result['z_score_interpretation']})")
            print(f"  Percentile: {result['percentile_in_baseline']:.1f}%")
            print(f"  Best Match: {result['best_match_second']:.2f}s (frame {result['best_match_frame']})")
            print()
    
    if verbose:
        # Summary
        anomaly_count = sum(1 for r in results if r["is_anomaly"])
        strong_anomaly_count = sum(1 for r in results if r["z_score"] < -2)
        
        print(f"{'='*80}")
        print(f"📊 SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {len(results)}")
        print(f"Anomalies detected: {anomaly_count} ({anomaly_count/len(results)*100:.1f}%)")
        print(f"Strong anomalies: {strong_anomaly_count} ({strong_anomaly_count/len(results)*100:.1f}%)")
        print(f"Normal cases: {len(results)-anomaly_count} ({(len(results)-anomaly_count)/len(results)*100:.1f}%)")
        print(f"{'='*80}\n")
    
    return results

# =========================
# 9️⃣ ROC Curve Analysis
# =========================
def analyze_threshold(labeled_data, method='statistical'):
    """
    Analyze different thresholds using labeled data.
    
    Args:
        labeled_data (list): [(text, second, is_anomaly_label), ...]
        method (str): Detection method
        
    Returns:
        dict: Optimal threshold and performance metrics
    """
    print("\n" + "="*80)
    print("🔬 THRESHOLD ANALYSIS")
    print("="*80)
    
    # Compute scores for all samples
    scores = []
    labels = []
    
    for text, second, label in labeled_data:
        result = detect_anomaly(text, second, method=method, threshold=0)  # No threshold filtering
        
        if method == 'statistical':
            score = -result['z_score']  # Higher = more anomalous
        else:
            score = result['anomaly_score']
        
        scores.append(score)
        labels.append(1 if label else 0)  # 1 = anomaly, 0 = normal
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    optimal_threshold = thresholds_pr[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"Method: {method}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {precision[optimal_idx]:.4f}")
    print(f"  Recall: {recall[optimal_idx]:.4f}")
    print(f"  F1-score: {optimal_f1:.4f}")
    print("="*80)
    
    return {
        "method": method,
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "optimal_f1": optimal_f1,
        "optimal_precision": precision[optimal_idx],
        "optimal_recall": recall[optimal_idx],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds_roc": thresholds_roc.tolist()
    }

# =========================
# 🔟 Example Usage
# =========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 INFERENCE EXAMPLES")
    print("="*80)
    
    # Test cases with ground truth labels
    test_cases = [
        # (text, target_second, is_anomaly)
        
        # Normal cases
        ("forklift approaching the cylindrical load on the left", 0, False),
        ("forklift lifting the cylindrical load upward", 4, False),
        ("forklift holding the load at maximum height", 9, False),
        ("forklift placing the load on the ground", 17, False),
        ("forklift moving backward away from the load", 20, False),
        
        # Anomaly cases - Wrong object
        ("person walking with a dog", 5, True),
        ("car driving on highway", 10, True),
        ("airplane flying overhead", 12, True),
        
        # Anomaly cases - Impossible actions
        ("forklift flying in the sky", 15, True),
        ("forklift swimming underwater", 8, True),
        
        # Anomaly cases - Wrong context
        ("person cooking in kitchen", 7, True),
        ("dog playing with ball", 14, True),
    ]
    
    # =========================
    # Example 1: Single Detection
    # =========================
    print("\n" + "="*80)
    print("📋 EXAMPLE 1: SINGLE DETECTION")
    print("="*80)
    
    result = detect_anomaly(
        "forklift lifting the cylindrical load upward", 
        target_second=4,
        method='statistical'
    )
    
    print(f"Text: {result['text']}")
    print(f"Target: {result['target_second']}s")
    print(f"Status: {'⚠️ ANOMALY' if result['is_anomaly'] else '✅ NORMAL'}")
    print(f"Z-score: {result['z_score']:.4f} ({result['z_score_interpretation']})")
    print(f"Max similarity: {result['max_similarity']:.4f}")
    print(f"Percentile: {result['percentile_in_baseline']:.1f}%")
    
    # =========================
    # Example 2: Batch Detection (Statistical)
    # =========================
    print("\n" + "="*80)
    print("📋 EXAMPLE 2: BATCH DETECTION (STATISTICAL METHOD)")
    print("="*80)
    
    results_statistical = batch_detect_anomalies(
        [(text, second) for text, second, _ in test_cases],
        method='statistical',
        window_size=3
    )
    
    # =========================
    # Example 3: Batch Detection (Threshold)
    # =========================
    print("\n" + "="*80)
    print("📋 EXAMPLE 3: BATCH DETECTION (THRESHOLD METHOD)")
    print("="*80)
    
    results_threshold = batch_detect_anomalies(
        [(text, second) for text, second, _ in test_cases],
        method='threshold',
        threshold=0.25,
        window_size=3
    )
    
    # =========================
    # Example 4: Threshold Analysis
    # =========================
    print("\n" + "="*80)
    print("📋 EXAMPLE 4: OPTIMAL THRESHOLD ANALYSIS")
    print("="*80)
    
    threshold_analysis = analyze_threshold(test_cases, method='statistical')
    
    # =========================
    # Save Results
    # =========================
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    # Save individual results
    output_file = config.output_dir / "anomaly_detection_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "statistical_method": results_statistical,
            "threshold_method": results_threshold,
            "threshold_analysis": threshold_analysis,
            "baseline_statistics": {
                "mean": float(baseline_mean),
                "std": float(baseline_std),
                "threshold": float(baseline_threshold)
            }
        }, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Save summary
    summary_file = config.output_dir / "detection_summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("ANOMALY DETECTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_suffix}\n")
        f.write(f"Detection method: Statistical (Z-score)\n")
        f.write(f"Total tests: {len(test_cases)}\n\n")
        
        f.write("Baseline Statistics:\n")
        f.write(f"  Mean: {baseline_mean:.4f}\n")
        f.write(f"  Std: {baseline_std:.4f}\n")
        f.write(f"  Threshold: {baseline_threshold:.4f}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  ROC AUC: {threshold_analysis['roc_auc']:.4f}\n")
        f.write(f"  Optimal F1: {threshold_analysis['optimal_f1']:.4f}\n")
        f.write(f"  Optimal threshold: {threshold_analysis['optimal_threshold']:.4f}\n\n")
        
        anomaly_count = sum(1 for r in results_statistical if r["is_anomaly"])
        f.write(f"Detection Results:\n")
        f.write(f"  Anomalies: {anomaly_count}/{len(results_statistical)}\n")
        f.write(f"  Normal: {len(results_statistical)-anomaly_count}/{len(results_statistical)}\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*80)

    challenging_cases = [
    # Borderline anomalies (biraz alakalı ama yanlış)
    ("machinery in motion", 6, True),           # Generic
    ("vehicle transporting cargo", 10, True),   # Generic
    ("industrial operation", 8, True),          # Çok genel
    
    # Hard negatives (doğru ama belirsiz)
    ("forklift with object", 4, False),         # Belirsiz ama doğru
    ("forklift in operation", 9, False),        # Generic ama doğru
    ("equipment moving load", 12, False),       # Doğru ama generic
    ]

    results = batch_detect_anomalies(
        [(text, second) for text, second, _ in challenging_cases],
        method='statistical'
    )

    # ROC AUC hala 1.0 mı?
    analysis = analyze_threshold(challenging_cases, method='statistical')
    print(f"Challenging cases ROC AUC: {analysis['roc_auc']:.4f}")