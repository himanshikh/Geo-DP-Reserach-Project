import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch.nn.functional as F
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Distance-Based MIA on: {device.type.upper()}")

# ==============================================================================
# 1. PATHS TO YOUR NOISY EMBEDDINGS
# ==============================================================================
PATH_CASIA_NOISY = 'casia_training_8k_normalised_embeddings_MI_eps_5.npy' # MEMBERS
PATH_LFW_NOISY = 'reversed_noise_addition_epsilon5_lfw.npy'     # NON-MEMBERS

if not os.path.exists(PATH_CASIA_NOISY) or not os.path.exists(PATH_LFW_NOISY):
    raise FileNotFoundError("Could not find the .npy files! Please ensure they are uploaded to the Colab sidebar.")

# ==============================================================================
# 2. LOAD & NORMALIZE EMBEDDINGS
# ==============================================================================
print("\n[1/4] Loading Noisy Embeddings...")
# Load numpy arrays
casia_np = np.load(PATH_CASIA_NOISY).astype(np.float32)
lfw_np = np.load(PATH_LFW_NOISY).astype(np.float32)

# Convert to PyTorch tensors and move to GPU for blazingly fast matrix math
casia_tensor = torch.tensor(casia_np, device=device)
lfw_tensor = torch.tensor(lfw_np, device=device)

# Ensure they are L2-Normalized (Cosine Similarity becomes simple Dot Product)
casia_norm = F.normalize(casia_tensor, p=2, dim=1)
lfw_norm = F.normalize(lfw_tensor, p=2, dim=1)

print(f"-> CASIA Members Loaded    : {casia_norm.shape[0]} vectors")
print(f"-> LFW Non-Members Loaded  : {lfw_norm.shape[0]} vectors")

# ==============================================================================
# 3. CALCULATE MAX COSINE SIMILARITIES (CHUNKED TO PREVENT OOM CRASHES)
# ==============================================================================
print("\n[2/4] Computing Cosine Similarities (Batched Memory-Safe Mode)...")

# Reduced chunk size drastically to prevent 6GB+ temporary matrices on 376k DB
BATCH_SIZE = 256

# --- NON-MEMBER SCORES (LFW vs CASIA) ---
print("-> Scoring Non-Members (LFW)...")
non_member_scores = []
with torch.no_grad():
    for i in tqdm(range(0, lfw_norm.size(0), BATCH_SIZE)):
        end = min(i + BATCH_SIZE, lfw_norm.size(0))
        chunk = lfw_norm[i:end]

        # Calculate chunk vs entire database
        sims = torch.mm(chunk, casia_norm.T)

        # Extract maximum similarity score
        max_sims = torch.max(sims, dim=1)[0]
        non_member_scores.extend(max_sims.cpu().tolist())

        # Free VRAM explicitly
        del sims
        torch.cuda.empty_cache()

non_member_scores = np.array(non_member_scores)

# --- MEMBER SCORES (CASIA vs CASIA) ---
print("-> Scoring Members (CASIA)...")
member_scores = []
with torch.no_grad():
    for i in tqdm(range(0, casia_norm.size(0), BATCH_SIZE)):
        end = min(i + BATCH_SIZE, casia_norm.size(0))
        chunk = casia_norm[i:end]

        # Calculate chunk vs entire database
        sims = torch.mm(chunk, casia_norm.T)

        # IMPORTANT: Mask the diagonal (comparing a vector exactly to itself)
        # Since we are chunking, row `r` inside the chunk corresponds to index `i + r` in the DB
        r_indices = torch.arange(end - i, device=device)
        c_indices = torch.arange(i, end, device=device)
        sims[r_indices, c_indices] = -2.0

        # Extract maximum similarity score against OTHER members
        max_sims = torch.max(sims, dim=1)[0]
        member_scores.extend(max_sims.cpu().tolist())

        # Free VRAM explicitly
        del sims
        torch.cuda.empty_cache()

member_scores = np.array(member_scores)

# ==============================================================================
# 4. CALCULATE MIA METRICS (TPR, FPR, ADVANTAGE, AUC)
# ==============================================================================
print("\n[3/4] Calculating MIA Advantage and ROC-AUC...")

# Labels: 1 for Member (CASIA), 0 for Non-Member (LFW)
y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
y_scores = np.concatenate([member_scores, non_member_scores])

# Calculate ROC Curve across all possible thresholds (t)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc_score = auc(fpr, tpr)

# Calculate Advantage = TPR - FPR
advantage = tpr - fpr
max_adv_idx = np.argmax(advantage)
max_adv = advantage[max_adv_idx]
optimal_threshold = thresholds[max_adv_idx]

# Extract specific metrics at the Optimal Threshold
best_tpr = tpr[max_adv_idx]
best_fpr = fpr[max_adv_idx]

print("\n" + "="*60)
print("MEMBERSHIP INFERENCE ATTACK (MIA) EVALUATION")
print("="*60)
print(f"Optimal Threshold (t) : {optimal_threshold:.4f}")
print(f"True Positives (TPR)  : {best_tpr:.4f} (Members correctly identified)")
print(f"False Positives (FPR) : {best_fpr:.4f} (Non-Members falsely flagged)")
print(f"Max Advantage (Adv)   : {max_adv:.4f}  <-- (0 = Perfect Privacy)")
print(f"ROC AUC Score         : {auc_score:.4f}  <-- (0.5 = Random Guess)")
print("="*60)

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
print("\n[4/4] Generating Plots...")
plt.figure(figsize=(14, 6))

# Subplot 1: Score Distributions
plt.subplot(1, 2, 1)
plt.hist(member_scores, bins=50, alpha=0.6, color='blue', label='Members (CASIA)', density=True)
plt.hist(non_member_scores, bins=50, alpha=0.6, color='red', label='Non-Members (LFW)', density=True)
plt.axvline(optimal_threshold, color='black', linestyle='dashed', linewidth=2.5, label=f'Optimal t={optimal_threshold:.3f}')

# Scaled up title, axes, and ticks
plt.title("Cosine Similarity Score Distribution ($s$)", fontsize=18, fontweight='bold', pad=15)
plt.xlabel("Max Cosine Similarity to Database", fontsize=18, fontweight='bold', labelpad=10)
plt.ylabel("Density", fontsize=18, fontweight='bold', labelpad=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: ROC Curve & Advantage
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'MIA ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--', label='Random Guessing (AUC = 0.5)')
plt.scatter([best_fpr], [best_tpr], color='red', s=150, zorder=5, label=f'Max Adv: {max_adv:.3f}\n(TPR: {best_tpr:.2f}, FPR: {best_fpr:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Scaled up title, axes, and ticks
plt.title('MIA Receiver Operating Characteristic (ROC)', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('False Positive Rate (FPR)', fontsize=18, fontweight='bold', labelpad=10)
plt.ylabel('True Positive Rate (TPR)', fontsize=18, fontweight='bold', labelpad=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14, loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("MIA_LFW_results.png", dpi=300)
plt.show()

print("\nEvaluation Complete! Plot saved as 'MIA_LFW_results.png'.")
