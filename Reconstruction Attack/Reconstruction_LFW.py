# ==============================================================================
# 0. ENVIRONMENT SETUP & DEPENDENCIES
# ==============================================================================
import sys
import os
import cv2
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
import numpy as np

# Clean progress bars for Jupyter
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import insightface
from insightface.app import FaceAnalysis

# ==============================================================================
# 1. A100 GPU HARDWARE OPTIMIZATIONS
# ==============================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Ampere-specific optimizations (A100 Tensor Cores)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Deploying A100-Optimized Random-Split Attack on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==============================================================================
# 2. DOWNLOAD LFW DATASET & ALIGNMENT
# ==============================================================================
print("\n[1/5] Downloading LFW dataset and aligning with metadata...")

PATH_LFW_EMBEDDINGS = 'reversed_noise_addition_epsilon12_lfw.npy'
PATH_LFW_LABELS = 'lfw_labels_min2.npy'

if not os.path.exists(PATH_LFW_EMBEDDINGS) or not os.path.exists(PATH_LFW_LABELS):
    raise FileNotFoundError(f"Ensure '{PATH_LFW_EMBEDDINGS}' and '{PATH_LFW_LABELS}' are in the Jupyter directory.")

stolen_embeddings = np.load(PATH_LFW_EMBEDDINGS).astype(np.float32)
target_labels = np.load(PATH_LFW_LABELS)

lfw_dataset = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
lfw_images = lfw_dataset.images
lfw_labels = lfw_dataset.target
lfw_target_names = lfw_dataset.target_names

print(f"LFW Source downloaded: {len(lfw_images)} images found.")

def normalize_name(name_str):
    return str(name_str).replace('_', ' ').strip().lower()

lfw_label_to_images = {}
for idx, target_int in enumerate(lfw_labels):
    norm_name = normalize_name(lfw_target_names[target_int])
    if norm_name not in lfw_label_to_images:
        lfw_label_to_images[norm_name] = []
    lfw_label_to_images[norm_name].append(lfw_images[idx])

lfw_image_usage_counters = {lbl: 0 for lbl in lfw_label_to_images.keys()}
aligned_images = []
missing_count = 0

for required_label in target_labels:
    norm_req = normalize_name(required_label)
    if norm_req in lfw_label_to_images:
        usage_idx = lfw_image_usage_counters[norm_req]
        available_imgs = lfw_label_to_images[norm_req]
        img = available_imgs[usage_idx % len(available_imgs)]
        aligned_images.append(img)
        lfw_image_usage_counters[norm_req] += 1
    else:
        missing_count += 1
        aligned_images.append(np.zeros((125, 94, 3), dtype=np.float32))

aligned_images = np.array(aligned_images)

if missing_count > 0:
    print(f"[!] Warning: {missing_count} missing labels padded with zeros.")
else:
    print("[+] SUCCESS: 100% of LFW embeddings perfectly matched to real images!")

EMBEDDING_DIM = stolen_embeddings.shape[1]
print(f"Detected Embedding Dimension: {EMBEDDING_DIM}")

# ==============================================================================
# 3. DATASET FORMATTING & RANDOM IMAGE-WISE SPLITTING (MODIFIED)
# ==============================================================================
print("\n[2/5] Preparing PyTorch Datasets (Random Image-Wise Split)...")

class LFWInversionDataset(Dataset):
    def __init__(self, embeddings_array, images_array, labels_array):
        self.embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        self.unique_string_labels = np.unique(labels_array)
        self.str_to_int_map = {name: idx for idx, name in enumerate(self.unique_string_labels)}
        self.int_labels = torch.tensor([self.str_to_int_map[lbl] for lbl in labels_array], dtype=torch.long)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
        ])
        
        self.images = []
        for idx in tqdm(range(len(images_array)), desc="Processing LFW Images"):
            img_np = images_array[idx]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            img_tensor = self.transform(img_np)
            self.images.append(img_tensor)
            
        self.images = torch.stack(self.images)

    def __len__(self): 
        return len(self.embeddings)
        
    def __getitem__(self, idx): 
        return self.embeddings[idx], self.images[idx], self.int_labels[idx]

# Initialize dataset
full_dataset = LFWInversionDataset(stolen_embeddings, aligned_images, target_labels)
total_images = len(full_dataset)

# --- NEW: RANDOM IMAGE-WISE SPLITTING INDICES ---
indices = np.arange(total_images)
np.random.seed(42)  # Maintain reproducibility
np.random.shuffle(indices)

# Calculated cutoff for strict 80/20 image split
train_cutoff = int(0.8 * total_images)
train_indices = indices[:train_cutoff]
test_indices = indices[train_cutoff:]

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# High-Performance Dataloader settings for A100
BATCH_SIZE = 256 
NUM_WORKERS = 8  

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)

print(f"Total Combined Images   : {total_images}")
print(f"Random Training Images  : {len(train_indices)} (80%)")
print(f"Random Testing Images   : {len(test_indices)} (20%)")
print(f"DataLoader Config       : Batch Size = {BATCH_SIZE}, CPU Workers = {NUM_WORKERS}")

# ==============================================================================
# 4. ENHANCED HIGH-CAPACITY DECODER ARCHITECTURE
# ==============================================================================
class DeepResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DeepResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + residual)

class EnhancedTemplateInversionDecoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(EnhancedTemplateInversionDecoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, 512 * 7 * 7)
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            DeepResidualBlock(256)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            DeepResidualBlock(128)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            DeepResidualBlock(64)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

# ==============================================================================
# PERCEPTUAL LOSS NETWORK DEFINITION (VGG-16)
# ==============================================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) 
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]

# ==============================================================================
# 5. TRAINING WITH HYBRID LOSS FUNCTION (A100 TUNED)
# ==============================================================================
model = EnhancedTemplateInversionDecoder(embedding_dim=EMBEDDING_DIM).to(device)
perceptual_net = VGGPerceptualLoss().to(device).eval()

# Tuned LR for massive batch sizes
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

EPOCHS = 500 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
l1_criterion = nn.L1Loss()
scaler = torch.amp.GradScaler('cuda')

print(f"\n[3/5] Training Deeper Attack Model on A100 for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, leave=False)
    for embeds, imgs, _ in loop:
        embeds, imgs = embeds.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(embeds)
            loss_l1 = l1_criterion(outputs, imgs)
            
            real_features = perceptual_net(imgs)
            fake_features = perceptual_net(outputs)
            
            loss_perceptual = 0.0
            for rf, ff in zip(real_features, fake_features):
                loss_perceptual += F.mse_loss(ff, rf)
                
            total_loss = loss_l1 + (0.1 * loss_perceptual)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

    scheduler.step()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1:^3}/{EPOCHS}] | Avg Hybrid Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

print("Training Complete!")

# ==============================================================================
# 6. INITIALIZE ARCFACE ENGINE
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

    orig_bgr = cv2.resize(orig_uint8, (112, 112))[:, :, ::-1]
    recon_bgr = cv2.resize(recon_uint8, (112, 112))[:, :, ::-1]

    emb_orig = recognizer.get_feat(orig_bgr)
    emb_recon = recognizer.get_feat(recon_bgr)

    if emb_orig is None or emb_recon is None or len(emb_orig) == 0 or len(emb_recon) == 0:
        return 0.0

    emb_orig, emb_recon = emb_orig.flatten(), emb_recon.flatten()
    return float(np.dot(emb_orig, emb_recon) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_recon) + 1e-8))

# ==============================================================================
# 7. EXECUTION & VISUALIZATION (RANDOM TEST SET - WITH MATPLOTLIB FP32 FIX)
# ==============================================================================
print("\n[5/5] Executing Attack on Randomly Split TEST Images and generating plots...")
model.eval()

dataset_to_plot = test_dataset
num_images = len(dataset_to_plot)
plot_batch_size = 5 

print(f"Total random test images to evaluate: {num_images}")

for start_idx in range(0, num_images, plot_batch_size):
    end_idx = min(start_idx + plot_batch_size, num_images)
    current_batch_size = end_idx - start_idx

    selected_embeds, selected_imgs, selected_labels = [], [], []

    for i in range(start_idx, end_idx):
        emb, img, lbl = dataset_to_plot[i]
        selected_embeds.append(emb)
        selected_imgs.append(img)
        selected_labels.append(lbl.item())

    test_embeds = torch.stack(selected_embeds).to(device)
    test_imgs = torch.stack(selected_imgs)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Keeps A100 output safe from the float16/bfloat16 Matplotlib bug
            recon_test_imgs = model(test_embeds).float().cpu()

    fig, axes = plt.subplots(2, current_batch_size, figsize=(3.6 * current_batch_size, 8))

    if current_batch_size == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"A100 Reconstruction on Randomly Split Test Images (Images {start_idx+1} to {end_idx})",
                 fontsize=14, fontweight='bold', y=1.02)

    orig_plot_np = test_imgs.permute(0, 2, 3, 1).numpy()
    recon_plot_np = recon_test_imgs.permute(0, 2, 3, 1).numpy()

    for i in range(current_batch_size):
        numeric_lbl = selected_labels[i]
        string_identity = full_dataset.unique_string_labels[numeric_lbl]

        # --- TOP ROW: Original Ground Truth ---
        axes[0, i].imshow(orig_plot_np[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Ground Truth\n{string_identity.title()}", fontweight='bold', pad=10, fontsize=10)

        # --- METRICS SYSTEM ---
        ind_ssim = ssim(orig_plot_np[i], recon_plot_np[i], data_range=1.0, channel_axis=-1)
        ind_psnr = psnr(orig_plot_np[i], recon_plot_np[i], data_range=1.0)
        ind_arc_sim = get_arcface_cossim(orig_plot_np[i], recon_plot_np[i], arcface_recognizer)

        # --- BOTTOM ROW: Reconstructed Attack Result ---
        axes[1, i].imshow(recon_plot_np[i])
        axes[1, i].axis('off')
        metric_text = r"$L_1 + \mathcal{L}_{percep}$ Recon" + f"\nSSIM: {ind_ssim:.2f} | PSNR: {ind_psnr:.1f}\nArcFace: {ind_arc_sim:.2f}"
        axes[1, i].set_title(metric_text, fontsize=10, fontweight='bold', pad=12)

    plt.subplots_adjust(hspace=0.45, wspace=0.1, top=0.88, bottom=0.05)
    plt.show()
    plt.close(fig)

print("\nFinished evaluating recovery attack on the randomized image split!")
