# ==========================================
# COLAB ENVIRONMENT AUTO-SETUP
# ==========================================
import sys
import os

if 'google.colab' in sys.modules:
    print("Google Colab detected! Installing required dependencies...")
    os.system('pip install -q insightface onnxruntime-gpu scikit-image')

import cv2
import gc
import zipfile
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import insightface
from insightface.app import FaceAnalysis

# ==============================================================================
# OPENCV & GPU STABILITY FIXES
# ==============================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Deploying FEI NBNet Attack on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==============================================================================
# 1. DATA EXTRACTION & PERFECT ALIGNMENT (USING IMAGEFOLDER)
# ==============================================================================
print("\n[1/5] Extracting FEI dataset and perfectly aligning to embeddings...")

# ---> UPDATE THESE FILENAMES IF NEEDED <---
PATH_FEI_ZIP = 'FEI_198_Clean_Identities.zip'
PATH_FEI_EMBEDDINGS = 'FEI_198_NoisyEmbeddings_eps_3.npy'

# Changed directory name to force a fresh, organized extraction
EXTRACT_DIR = 'fei_organized_images'

if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting and organizing {PATH_FEI_ZIP} into class folders...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    with zipfile.ZipFile(PATH_FEI_ZIP, 'r') as z:
        for file_path in z.namelist():
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()

            if not filename or ext not in valid_exts or filename.startswith('._'):
                continue

            # Parse identity from filename to create subfolders
            match = re.search(r'(\d+)', filename)
            identity = match.group(1) if match else "Unknown"

            identity_dir = os.path.join(EXTRACT_DIR, f"Person_{identity}")
            os.makedirs(identity_dir, exist_ok=True)

            target_path = os.path.join(identity_dir, filename)
            with z.open(file_path) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
    print(" Extraction and organization complete!")
else:
    print(f" Organized directory {EXTRACT_DIR} already exists. Skipping extraction.")

# Load stolen embeddings
stolen_embeddings = np.load(PATH_FEI_EMBEDDINGS).astype(np.float32)

# ==============================================================================
# 2. DATASET FORMATTING (RAM PRELOAD FOR SPEED)
# ==============================================================================
print("\n[2/5] Preparing PyTorch Datasets (Strict User-Wise Split)...")

class FEIInversionDataset(Dataset):
    def __init__(self, embeddings_array, extract_dir):
        self.embeddings = torch.tensor(embeddings_array, dtype=torch.float32)

        # USE NATIVE IMAGEFOLDER TO GUARANTEE PERFECT ALIGNMENT WITH EMBEDDINGS
        image_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor() # Automatically converts to [0, 1] RGB Tensor
        ])

        print("⚡ Loading Images to RAM via strict ImageFolder ordering...")
        self.raw_dataset = datasets.ImageFolder(root=extract_dir, transform=image_transform)

        if len(self.raw_dataset) != len(self.embeddings):
            print(f"\n[!] CRITICAL WARNING: {len(self.raw_dataset)} images but {len(self.embeddings)} embeddings!")
            print("Check your dataset folders. Exiting to prevent corrupt training.")
            exit()

        self.images = []
        self.labels = []

        for idx in tqdm(range(len(self.raw_dataset)), desc="Preloading to RAM"):
            img, label = self.raw_dataset[idx]
            self.images.append(img)
            self.labels.append(label)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx): return self.embeddings[idx], self.images[idx], self.labels[idx]

# Initialize dataset
full_dataset = FEIInversionDataset(stolen_embeddings, EXTRACT_DIR)
target_labels = full_dataset.labels.numpy()

unique_users = np.unique(target_labels)
np.random.seed(42)
np.random.shuffle(unique_users)

# 80% Train / 20% Test User Split
train_user_cutoff = int(0.8 * len(unique_users))
train_users = set(unique_users[:train_user_cutoff])
test_users = set(unique_users[train_user_cutoff:])

train_indices = [i for i, label in enumerate(target_labels) if label in train_users]
test_indices = [i for i, label in enumerate(target_labels) if label in test_users]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Total Unique Identities : {len(unique_users)}")
print(f"Hacker Trained On       : {len(train_users)} identities")
print(f"Hacker Attacking        : {len(test_users)} unseen identities")

# ==============================================================================
# 3. NBNet-INSPIRED DECODER ARCHITECTURE
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class TemplateInversionDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TemplateInversionDecoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True), ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        return self.decoder(x)

# ==============================================================================
# 4. TRAINING THE NBNet ATTACK
# ==============================================================================
model = TemplateInversionDecoder(embedding_dim=128).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 300
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

print(f"\n[3/5] Training Reconstruction Attack Model for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, leave=False)
    for embeds, imgs, _ in loop:
        embeds, imgs = embeds.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(embeds)
            loss = criterion(outputs, imgs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

    scheduler.step()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1:^3}/{EPOCHS}] | Avg MSE Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

print("Training Complete!")

# ==============================================================================
# 5. INITIALIZE ARCFACE
# ==============================================================================
torch.cuda.empty_cache()
gc.collect()

print("\n[4/5] Initializing ArcFace Analysis Pipeline for Evaluation...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
arcface_recognizer = app.models['recognition']

def get_arcface_cossim(orig_img_np, recon_img_np, recognizer):
    orig_uint8 = (np.clip(orig_img_np, 0, 1) * 255).astype(np.uint8)
    recon_uint8 = (np.clip(recon_img_np, 0, 1) * 255).astype(np.uint8)

    orig_bgr = cv2.resize(orig_uint8[:, :, ::-1], (112, 112))
    recon_bgr = cv2.resize(recon_uint8[:, :, ::-1], (112, 112))

    emb_orig = recognizer.get_feat(orig_bgr)
    emb_recon = recognizer.get_feat(recon_bgr)

    if emb_orig is None or emb_recon is None or len(emb_orig) == 0 or len(emb_recon) == 0:
        return 0.0

    emb_orig, emb_recon = emb_orig.flatten(), emb_recon.flatten()
    return np.dot(emb_orig, emb_recon) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_recon) + 1e-8)

def calculate_metrics(original, reconstructed, recognizer):
    total_ssim, total_psnr, total_px_sim, total_arc_sim = 0, 0, 0, 0
    orig_np = original.permute(0, 2, 3, 1).cpu().numpy()
    recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()

    for i in range(original.shape[0]):
        total_ssim += ssim(orig_np[i], recon_np[i], data_range=1.0, channel_axis=-1)
        total_psnr += psnr(orig_np[i], recon_np[i], data_range=1.0)
        orig_flat, recon_flat = orig_np[i].flatten(), recon_np[i].flatten()
        total_px_sim += np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-8)
        total_arc_sim += get_arcface_cossim(orig_np[i], recon_np[i], recognizer)

    return total_ssim, total_psnr, total_px_sim, total_arc_sim

# ==============================================================================
# 6. EXECUTION & PLOTTING (TEST SET ONLY - 4 IMAGES PER ROW)
# ==============================================================================
print("\n[5/5] Executing Attack on TEST Templates and plotting...")

model.eval()

# ----------------------------------------------------------
# IMPORTANT: Plot ONLY the test dataset
# ----------------------------------------------------------
dataset_to_plot = test_dataset

num_images = len(dataset_to_plot)
plot_batch_size = 4

print(f"Total TEST images: {num_images}")

for start_idx in range(0, num_images, plot_batch_size):

    end_idx = min(start_idx + plot_batch_size, num_images)
    current_batch_size = end_idx - start_idx

    selected_embeds = []
    selected_imgs = []
    selected_labels = []

    # ----------------------------------------------------------
    # Collect one batch from TEST SET
    # ----------------------------------------------------------
    for i in range(start_idx, end_idx):

        emb, img, lbl = dataset_to_plot[i]

        selected_embeds.append(emb)
        selected_imgs.append(img)
        selected_labels.append(lbl.item())

    test_embeds = torch.stack(selected_embeds).to(device)
    test_imgs = torch.stack(selected_imgs)

    # ----------------------------------------------------------
    # Reconstruction
    # ----------------------------------------------------------
    with torch.no_grad():
        recon_test_imgs = model(test_embeds).cpu()

    # ----------------------------------------------------------
    # Create Figure
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        current_batch_size,
        figsize=(16, 7),
        constrained_layout=True
    )

    if current_batch_size == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"NBNet Reconstruction on TEST Set\n"
        f"Images {start_idx+1} - {end_idx}",
        fontsize=16,
        fontweight="bold"
    )

    orig_plot_np = test_imgs.permute(0, 2, 3, 1).numpy()
    recon_plot_np = recon_test_imgs.permute(0, 2, 3, 1).numpy()

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    for i in range(current_batch_size):

        lbl = selected_labels[i]

        axes[0, i].imshow(orig_plot_np[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"Ground Truth\nIdentity: {lbl}",
            fontsize=11,
            fontweight="bold",
            pad=10
        )

        ind_ssim = ssim(
            orig_plot_np[i],
            recon_plot_np[i],
            data_range=1.0,
            channel_axis=-1
        )

        ind_psnr = psnr(
            orig_plot_np[i],
            recon_plot_np[i],
            data_range=1.0
        )

        ind_arc_sim = get_arcface_cossim(
            orig_plot_np[i],
            recon_plot_np[i],
            arcface_recognizer
        )

        metric_text = (
            r"$\epsilon_{3}$ Recon"
            f"\nSSIM: {ind_ssim:.2f} | PSNR: {ind_psnr:.1f}"
            f"\nArcFace: {ind_arc_sim:.2f}"
        )

        axes[1, i].imshow(recon_plot_np[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(
            metric_text,
            fontsize=10,
            fontweight="bold",
            pad=10
        )

    plt.show()
    plt.close(fig)

print("\nFinished plotting TEST images!")
