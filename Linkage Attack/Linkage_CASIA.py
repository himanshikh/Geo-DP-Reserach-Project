# Enable inline plotting in Jupyter Notebook
%matplotlib inline

import time
import math
import torch
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =====================================================================
# ANSI TERMINAL COLORS
# =====================================================================
C_DSYS = '\033[96m'  # Cyan for D_sys
C_ACC = '\033[92m'   # Green for Accuracies
C_RESET = '\033[0m'  # Reset

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
raw_epsilons = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 2, 2.2, 2.4, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12]
epsilons = sorted(list(set(raw_epsilons)))

# L2 Sensitivity for normalized vectors is exactly 2.0
delta2 = 2.0 
sigma_bases = [(delta2 * np.sqrt(2 * np.log(1.25 / 1e-5))) / eps for eps in epsilons]

# --- REVERSED HYBRID NOISE CONFIGURATION ---
TOP_K_NOISE = 20    # Number of dominant channels to apply NORMAL (unbounded) noise
NOISE_BOUND = 0.1   # Clip limit for the REMAINING channels

print("=" * 115)
print(f"--- FEI 198 128D DP-SECURE EVAL: REVERSED HYBRID NOISE (ALL PAIRS) ---")
print("=" * 115)
t_total = time.time()

# =====================================================================
# 2. ISO UNLINKABILITY METRIC FUNCTION (From evaluateUnlinkability.py)
# =====================================================================
def calculate_iso_unlinkability(matedScores, nonMatedScores):
    """
    Computes D_sys and local linkability D(s) using the exact methodology 
    from Gomez-Barrero et al. (IS16, TIFS17).
    """
    # 1. Plot score distributions hiddenly to extract KDE coordinates matching Seaborn exactly
    fig, temp_ax = plt.subplots()
    sns.kdeplot(matedScores, ax=temp_ax, clip=(-1.0, 1.0))
    x1, y1 = temp_ax.get_lines()[0].get_data()
    sns.kdeplot(nonMatedScores, ax=temp_ax, clip=(-1.0, 1.0))
    x2, y2 = temp_ax.get_lines()[1].get_data()
    plt.close(fig)

    # 2. Define ranges
    bin_centers_common = np.linspace(max([x1.min(), x2.min()]), min([x1.max(), x2.max()]), num=128, endpoint=True)
    bin_centers = np.linspace(min([x1.min(), x2.min()]), max([x1.max(), x2.max()]), num=128, endpoint=True)

    # Determine the range of common scores for tail additions
    index = np.where(bin_centers > bin_centers_common[0])
    ini = index[0][0] if len(index[0]) > 0 else 0
    
    index = np.where(bin_centers > bin_centers_common[-1])
    fin = index[0][0] if len(index[0]) > 0 else 128

    # 3. Interpolate values
    y1values = interp1d(x1, y1, kind='cubic', bounds_error=False, fill_value=0.0)
    y1_c = y1values(bin_centers_common)
    COEF = sum(y1_c)
    if COEF != 0: y1_c = y1_c / COEF

    y2values = interp1d(x2, y2, kind='cubic', bounds_error=False, fill_value=0.0)
    y2_c = y2values(bin_centers_common)
    if COEF != 0: y2_c = y2_c / COEF

    # 4. Compute Likelihood Ratio and Local Linkability D(s)
    with np.errstate(divide='ignore', invalid='ignore'):
        LR = np.divide(y1_c, y2_c)
    
    D = 2*(LR/(1 + LR)) - 1
    D[LR <= 1] = 0
    D[y2_c == 0] = 1
    D[y1_c == 0] = 0
    D[np.isnan(D)] = 0

    # 5. Find values for whole domain by setting tails to 0
    a = np.argwhere(bin_centers >= x1[0])
    b = np.argwhere(bin_centers <= x1[-1])
    bin_centers_1 = bin_centers[np.intersect1d(a, b)]

    a = np.argwhere(bin_centers >= x2[0])
    b = np.argwhere(bin_centers <= x2[-1])
    bin_centers_2 = bin_centers[np.intersect1d(a, b)]
    
    y1_full = np.zeros(len(bin_centers))
    y1_full[np.intersect1d(np.argwhere(bin_centers >= x1[0]), np.argwhere(bin_centers <= x1[-1]))] = y1values(bin_centers_1)

    # 6. Extrapolate D to the tails
    Dvalues = interp1d(bin_centers_common, D, kind='cubic', bounds_error=False, fill_value=0.0)
    D_ext = Dvalues(bin_centers[ini:fin])
    
    if x1.min() < x2.min():
        D_ext = np.insert(D_ext, 0, np.ones(ini))
    else:
        D_ext = np.insert(D_ext, 0, np.zeros(ini))

    if x1.max() > x2.max():
        pad_len = 128 - fin
        if pad_len > 0: D_ext = np.insert(D_ext, len(D_ext), np.ones(pad_len))
    else:
        pad_len = 128 - fin
        if pad_len > 0: D_ext = np.insert(D_ext, len(D_ext), np.zeros(pad_len))

    # Format bounds
    D_ext = D_ext[:128]
    D_ext[D_ext < 0] = 0
    D_ext[D_ext > 1] = 1

    # 7. Compute Global System Linkability
    Dsys = np.trapz(x=bin_centers, y=D_ext * y1_full)
    
    return Dsys, bin_centers_1, bin_centers_2, bin_centers, D_ext, y1values, y2values


# =====================================================================
# 3. SETUP & DATA LOADING
# =====================================================================
print("1. Loading FEI 198 Clean 128D PCA/LDA Normalized Embeddings...")

EMBEDDINGS_FILE = 'FEI_198_Clean_128dNormalised_casia_vgg_embeddings.npy'
LABELS_FILE = 'FEI_198_Clean_128dNormalised_casia_vgg_labels.npy'
LDA_MODEL_FILE = 'lda_integrated_casia_vgg.pkl'

try:
    X_clean = np.load(EMBEDDINGS_FILE).astype(np.float32)
    y = np.load(LABELS_FILE)
    lda = joblib.load(LDA_MODEL_FILE)
except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Please check your file paths and try again.")
    exit()

num_images, d = X_clean.shape
lam = lda.explained_variance_ratio_[:d] 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unique_names, y_ints = np.unique(y, return_inverse=True)
y_tensor = torch.tensor(y_ints, dtype=torch.long, device=device)


# =====================================================================
# 4. PRE-COMPUTE MATED & NON-MATED INDICES (ALL PAIRS)
# =====================================================================
print("2. Pre-computing ALL mated/non-mated pairs...")

i_idx, j_idx = np.triu_indices(num_images, k=1)
genuine_mask = (y_ints[i_idx] == y_ints[j_idx])
impostor_mask = ~genuine_mask

mated_1, mated_2 = i_idx[genuine_mask], j_idx[genuine_mask]
NUM_GENUINE = len(mated_1)
imp_1, imp_2 = i_idx[impostor_mask], j_idx[impostor_mask]
NUM_IMPOSTOR = len(imp_1)

print(f"   -> Total Genuine Pairs Configured : {NUM_GENUINE:,}")
print(f"   -> Total Impostor Pairs Configured: {NUM_IMPOSTOR:,}")

np.random.seed(42)
nm1 = np.random.randint(0, num_images, 100000)
nm2 = np.random.randint(0, num_images, 100000)
mask = y_ints[nm1] != y_ints[nm2]
nm1_dsys, nm2_dsys = nm1[mask][:10000], nm2[mask][:10000]

m_idx_dsys = np.arange(num_images) 

results = {}
plot_data = {}
global_dsys = []

# =====================================================================
# 5. MAIN EVALUATION LOOP
# =====================================================================
print("\n3. Commencing Reversed Hybrid Subspace Privacy Tests on FEI 198...")
print("-" * 145)
print(f"{'Eps':<6} | {'Mean(Min/Max) Gen':<25} | {'Mean(Min/Max) Imp':<25} | {'Verif Acc%':<12} | {'EER%':<8} | {'Rank-1%':<10} | {'Dsys':<15}")
print("-" * 145)

thresholds_verif = np.linspace(-1.0, 1.0, 5000)

for eps in epsilons:
    sig_base = (delta2 * np.sqrt(2 * np.log(1.25 / 1e-5))) / eps
    sigmas_all = sig_base * (lam / lam[0])**2
    
    # -----------------------------------------------------------------
    # PART A: ACCURACY & VERIFICATION EVALUATION (Database A)
    # -----------------------------------------------------------------
    np.random.seed(int(eps * 100) + 1)
    noise_matrix_A = np.zeros_like(X_clean, dtype=np.float32)
    raw_noise_A = np.random.normal(0, sigmas_all, size=(num_images, d)).astype(np.float32)
    
    noise_matrix_A[:, :TOP_K_NOISE] = raw_noise_A[:, :TOP_K_NOISE]
    noise_matrix_A[:, TOP_K_NOISE:] = np.clip(raw_noise_A[:, TOP_K_NOISE:], -NOISE_BOUND, NOISE_BOUND)
    
    X_tilde_A = (X_clean + noise_matrix_A)
    DB1 = X_tilde_A / (np.linalg.norm(X_tilde_A, axis=1, keepdims=True) + 1e-8)
    
    # --- IDENTIFICATION (RANK-1, 5, 10) ---
    X_norm = torch.tensor(DB1, dtype=torch.float32, device=device)
    correct_1, correct_5, correct_10 = 0, 0, 0
    batch_size = 1000 
    
    for i in range(0, num_images, batch_size):
        end = min(i + batch_size, num_images)
        sims = torch.mm(X_norm[i:end], X_norm.T)
        sims[torch.arange(end-i), torch.arange(i, end, device=device)] = -2.0
        
        topk_vals, topk_indices = torch.topk(sims, k=10, dim=1)
        topk_preds = y_tensor[topk_indices] 
        true_labels = y_tensor[i:end].unsqueeze(1) 
        
        matches = (topk_preds == true_labels)
        correct_1 += torch.sum(matches[:, 0]).item()
        correct_5 += torch.sum(matches[:, :5].any(dim=1)).item()
        correct_10 += torch.sum(matches.any(dim=1)).item()
        
        del sims, topk_vals, topk_indices, topk_preds, matches
        torch.cuda.empty_cache()
        
    acc_1 = (correct_1 / num_images) * 100
    acc_5 = (correct_5 / num_images) * 100
    acc_10 = (correct_10 / num_images) * 100

    # --- VERIFICATION ACCURACY & EER (1:1) ---
    sim_matrix_A = np.dot(DB1, DB1.T)
    sim_matrix_A = np.clip(sim_matrix_A, -1.0, 1.0) 
    
    g_scores_verif = sim_matrix_A[mated_1, mated_2]
    i_scores_verif = sim_matrix_A[imp_1, imp_2]
    
    stats_g = f"{np.mean(g_scores_verif):.2f}({np.min(g_scores_verif):.2f}/{np.max(g_scores_verif):.2f})"
    stats_i = f"{np.mean(i_scores_verif):.2f}({np.min(i_scores_verif):.2f}/{np.max(i_scores_verif):.2f})"
    
    g_sorted = np.sort(g_scores_verif)
    i_sorted = np.sort(i_scores_verif)
    
    tp_v_arr = NUM_GENUINE - np.searchsorted(g_sorted, thresholds_verif)
    fp_v_arr = NUM_IMPOSTOR - np.searchsorted(i_sorted, thresholds_verif)
    tn_v_arr = NUM_IMPOSTOR - fp_v_arr
    fn_v_arr = NUM_GENUINE - tp_v_arr
    
    acc_arr = ((tp_v_arr + tn_v_arr) / (NUM_GENUINE + NUM_IMPOSTOR)) * 100
    best_verif_acc = np.max(acc_arr)
    
    fmr_v_arr = fp_v_arr / NUM_IMPOSTOR
    fnmr_v_arr = fn_v_arr / NUM_GENUINE
    
    eer_idx = np.argmin(np.abs(fmr_v_arr - fnmr_v_arr))
    eer_val = (fmr_v_arr[eer_idx] + fnmr_v_arr[eer_idx]) / 2.0
    
    # -----------------------------------------------------------------
    # PART B: ISO D_sys EVALUATION (Database B)
    # -----------------------------------------------------------------
    np.random.seed(int(eps * 100) + 2)
    noise_matrix_B = np.zeros_like(X_clean, dtype=np.float32)
    raw_noise_B = np.random.normal(0, sigmas_all, size=(num_images, d)).astype(np.float32)
    noise_matrix_B[:, :TOP_K_NOISE] = raw_noise_B[:, :TOP_K_NOISE]
    noise_matrix_B[:, TOP_K_NOISE:] = np.clip(raw_noise_B[:, TOP_K_NOISE:], -NOISE_BOUND, NOISE_BOUND)
    
    X_tilde_B = (X_clean + noise_matrix_B)
    DB2 = X_tilde_B / (np.linalg.norm(X_tilde_B, axis=1, keepdims=True) + 1e-8)
    
    g_scores_cross = np.clip(np.sum(DB1[m_idx_dsys] * DB2[m_idx_dsys], axis=1), -1.0, 1.0)
    i_scores_cross = np.clip(np.sum(DB1[nm1_dsys] * DB2[nm2_dsys], axis=1), -1.0, 1.0)
    
    try:
        true_dsys, bc1, bc2, bc, D_ext, y1v, y2v = calculate_iso_unlinkability(g_scores_cross, i_scores_cross)
    except Exception:
        true_dsys = 1.0
        bc1, bc2, bc, D_ext, y1v, y2v = (None,)*6
        
    global_dsys.append(true_dsys)
        
    results[float(eps)] = (best_verif_acc, eer_val * 100, acc_1, acc_5, acc_10, true_dsys)
    
    if float(eps) in [3.0, 5.0, 12.0]:
        plot_data[float(eps)] = {
            'bc1': bc1, 'bc2': bc2, 'bc': bc,
            'D': D_ext, 'y1v': y1v, 'y2v': y2v,
            'dsys': true_dsys, 'sig_base': sig_base
        }
    
    print(f"{eps:<6.1f} | {stats_g:<25} | {stats_i:<25} | {C_ACC}{best_verif_acc:<12.4f}{C_RESET} | {eer_val*100:<8.4f} | {C_ACC}{acc_1:<10.4f}{C_RESET} | {C_DSYS}{true_dsys:<15.4f}{C_RESET}")

print("-" * 145)


# =====================================================================
# 6. MATPLOTLIB VISUALIZATION (ISO STANDARD distributions.eps STYLE)
# =====================================================================
print("\n4. Generating Linkability Visualization for Epsilon 3, 5, 12...")

sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.5})
sns.set_style("white")

plot_epsilons = [3.0, 5.0, 12.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, eps in enumerate(plot_epsilons):
    ax = axes[idx]
    ax2 = ax.twinx()
    
    data = plot_data[float(eps)]
    bc1, bc2, bc = data['bc1'], data['bc2'], data['bc']
    D, y1v, y2v = data['D'], data['y1v'], data['y2v']
    dsys = data['dsys']
    
    # MODIFIED: Red non-mated line is now solid (linestyle='-'), Black local linkability line
    lns1 = ax.plot(bc1, y1v(bc1), label='Mated', color=sns.xkcd_rgb["medium green"])
    lns2 = ax.plot(bc2, y2v(bc2), label='Non-Mated', color=sns.xkcd_rgb["pale red"], linestyle='-', linewidth=3)
    lns3 = ax2.plot(bc, D, label='$\mathrm{D}_{\leftrightarrow}(s)$', color='black', linewidth=4)

    ax.set_xlim([-1.0, 1.0])
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Probability Density" if idx == 0 else "")
    ax.set_xlabel("Cosine Similarity(s)")
    
    title_text = f"$\\epsilon={eps}$\n$D_{{sys}}$ = {dsys:.4f}"
    ax.set_title(title_text, pad=15)
    
    # Render legend on ALL subplots
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')

    # Dynamically scale Y limits
    max_y = max(max(y1v(bc1) if len(bc1)>0 else [0]), max(y2v(bc2) if len(bc2)>0 else [0]))
    ax.set_ylim([0, max_y + 0.05 * max_y])
    
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("$\mathrm{D}_{\leftrightarrow}(s)$" if idx == 2 else "")

plt.tight_layout()

# MODIFIED: Save the image with the specific filename requested
plt.savefig('linkage_attack_fei.png', dpi=300, bbox_inches='tight')

plt.show() 
plt.close()

# Global D_sys Curve
print("\n5. Generating Global System Linkability vs Epsilon curve...")
plt.figure(figsize=(10, 6))
plt.plot(epsilons, global_dsys, marker='D', linestyle='-', color='darkred', linewidth=2, markersize=6)
plt.fill_between(epsilons, global_dsys, color='darkred', alpha=0.1)

plt.title("FEI 198 Cross-Database System Linkability ($D_{sys}$) vs Privacy Budget ($\epsilon$)", fontsize=14, fontweight='bold')
plt.xlabel("Privacy Budget ($\epsilon$)", fontsize=12)
plt.ylabel("System Linkability ($D_{sys}$)", fontsize=12)
plt.xlim([0, max(epsilons) + 0.5])
plt.ylim([0, 1.05])
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show() 
plt.close()

print(f"\nTotal execution time: {time.time() - t_total:.2f} seconds")
