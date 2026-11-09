import os
import glob
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from insightface.app import FaceAnalysis

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZIP_PATH = "casia_2k_ready.zip"               
EXTRACT_DIR = "./casia_extracted"
LABELS_PATH = "casia_2k_ordered_labels.npy"                   
EMBEDDINGS_PATH = "casia_2k_labelled_NoisyEmbeddings_ep12.npy"

BATCH_SIZE = 64                              # To prevent mode collapse
INITIAL_LEARNING_RATE = 2e-4
EPOCHS = 200
print(f"Using execution device: {DEVICE}")

# ==========================================
# 2. DATA EXTRACTION & STRICT 1:1 ALIGNMENT
# ==========================================

if not os.path.exists(EXTRACT_DIR) and os.path.exists(ZIP_PATH):
    print("Extracting images...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Load raw labels and embeddings

labels = np.load(LABELS_PATH, allow_pickle=True)

embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)



# ---> CRITICAL FIX FOR 0-D ARRAYS <---

if isinstance(labels, np.ndarray) and labels.ndim == 0:

    labels = labels.item()

if isinstance(embeddings, np.ndarray) and embeddings.ndim == 0:

    embeddings = embeddings.item()



# Extract data safely if they are stored as dictionaries

if isinstance(labels, dict): 

    labels = list(labels.values())[0]

if isinstance(embeddings, dict):

    key = next((k for k in embeddings.keys() if 'embed' in k.lower()), list(embeddings.keys())[0])

    embeddings = embeddings[key]



# Convert labels to clean flat list of strings

labels_flat = np.array(labels).flatten().astype(str).tolist()



# Convert embeddings safely into a proper 2D float32 numpy array

if isinstance(embeddings, (list, tuple)):

    embeddings = np.stack(embeddings).astype(np.float32)

elif isinstance(embeddings, np.ndarray):

    if embeddings.dtype == object:

        embeddings = np.stack(embeddings).astype(np.float32)

    else:

        embeddings = embeddings.astype(np.float32)



# Build a fast lookup dictionary of ALL extracted image files

all_found_images = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.*"), recursive=True)

img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in all_found_images}



# Only keep data where the label string perfectly pairs with an active image filename

aligned_paths, aligned_labels, aligned_embeddings = [], [], []



for idx, lbl in enumerate(labels_flat):

    if lbl in img_dict:

        aligned_paths.append(img_dict[lbl])

        aligned_labels.append(lbl)

        aligned_embeddings.append(embeddings[idx])



image_paths = aligned_paths

labels_flat = aligned_labels

embeddings = np.array(aligned_embeddings)



assert len(image_paths) == len(labels_flat) == len(embeddings), "Strict alignment sanity count failed!"

print(f"Strict Alignment Complete: {len(image_paths)} perfect image-to-embedding pairs found.")



# ==========================================

# 3. PYTORCH DATASET DEFINITION

# ==========================================

class CasiaReconstructionDataset(Dataset):

    def __init__(self, data_list):

        self.data = data_list



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_path, embedding, label = self.data[idx]

        

        # Load image as RGB

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (112, 112))

        

        # Normalize to [-1, 1] matching Tanh output

        img = (img / 127.5) - 1.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) # [C, H, W]

        emb_tensor = torch.tensor(embedding, dtype=torch.float32)

        

        return emb_tensor, img, label



# ==========================================

# 4. RANDOM DATA SPLIT & DATALOADERS

# ==========================================

master_data = list(zip(image_paths, embeddings, labels_flat))



# Random Image-wise Split

train_data, test_data = train_test_split(master_data, test_size=0.2, random_state=42)

print(f"Splitting complete -> Training items: {len(train_data)} | Testing items: {len(test_data)}")



train_loader = DataLoader(CasiaReconstructionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

test_loader = DataLoader(CasiaReconstructionDataset(test_data), batch_size=1, shuffle=False)



# ==========================================

# 5. ARCHITECTURE & VGG LOSS

# ==========================================

class LargerNBNet_112(nn.Module):

    def __init__(self, embedding_dim=512):

        super(LargerNBNet_112, self).__init__()

        

        self.fc = nn.Sequential(

            nn.Linear(embedding_dim, 512 * 7 * 7),

            nn.BatchNorm1d(512 * 7 * 7),

            nn.LeakyReLU(0.2, inplace=True)

        )

        

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), 

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2, inplace=True),

            

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1), 

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2, inplace=True),

            

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),   

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2, inplace=True),

            

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),

            nn.BatchNorm2d(32),

            nn.LeakyReLU(0.2, inplace=True),

            

            nn.Conv2d(32, 3, kernel_size=3, padding=1),

            nn.Tanh()

        )



    def forward(self, x):

        x = self.fc(x)

        x = x.view(-1, 512, 7, 7)

        return self.decoder(x)



class VGGLoss(nn.Module):

    def __init__(self, device):

        super(VGGLoss, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.vgg_submodel = nn.Sequential(*list(vgg.children())[:16]).to(device)

        self.vgg_submodel.eval()

        

        for param in self.vgg_submodel.parameters():

            param.requires_grad = False

            

        self.criterion = nn.L1Loss()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)

        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)



    def forward(self, fake_img, real_img):

        fake_norm = (fake_img + 1.0) / 2.0

        real_norm = (real_img + 1.0) / 2.0

        fake_norm = (fake_norm - self.mean) / self.std

        real_norm = (real_norm - self.mean) / self.std

        

        fake_features = self.vgg_submodel(fake_norm)

        real_features = self.vgg_submodel(real_norm)

        

        return self.criterion(fake_features, real_features)



model = LargerNBNet_112(embedding_dim=embeddings.shape[1]).to(DEVICE)



# Initialize Losses

l1_criterion = nn.L1Loss()

vgg_criterion = VGGLoss(DEVICE)



optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, betas=(0.5, 0.999))

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)



# ==========================================

# 6. TRAINING EXECUTION LOOP

# ==========================================

print("\nStarting optimization loop with L1 + VGG Loss...")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for emb, img_real, _ in train_loader:

        emb, img_real = emb.to(DEVICE, non_blocking=True), img_real.to(DEVICE, non_blocking=True)

        

        optimizer.zero_grad()

        img_fake = model(emb)

        

        # Calculate L1 and VGG Loss

        loss_l1 = l1_criterion(img_fake, img_real)

        loss_vgg = vgg_criterion(img_fake, img_real)

        

        # Combine Losses (1.0 * L1 + 0.5 * VGG)

        total_loss = loss_l1 + (0.5 * loss_vgg)

        

        total_loss.backward()

        optimizer.step()

        

        running_loss += total_loss.item()

    

    scheduler.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Mean Total Loss: {running_loss/len(train_loader):.5f}")



torch.save(model.state_dict(), "nbnet_casia_reconstruction_fixed.pth")

print("Model saved as 'nbnet_casia_reconstruction_fixed.pth'")


# ==========================================
# 7. EVALUATION & ARCFACE ENGINE INITIALIZATION
# ==========================================
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

recognizer = app.models['recognition']

def get_arcface_embedding(img_rgb_uint8):
    # InsightFace's recognition model expects a BGR image
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    
    try:
        # Bypass the detector and directly extract the 512-D embedding
        # get_feat() returns an array of shape (1, 512), so we take index [0]
        embedding = recognizer.get_feat(img_bgr)
        return embedding[0]
    except Exception as e:
        return None

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# ==========================================
# 8. TESTING & FOOLPROOF PLOTTING GRID
# ==========================================
print("\nEvaluating test set and rendering visualization grid...")
model.eval()

unique_users_collected = []
plotted_ids = set()

orig_detected, fake_detected = 0, 0
total_checked = 0

for emb, img_real, label in test_loader:
    lbl_val = str(label[0]) if isinstance(label, (list, tuple, torch.Tensor)) else str(label)
    
    if lbl_val in plotted_ids:
        continue
        
    emb, img_real = emb.to(DEVICE), img_real.to(DEVICE)
    with torch.no_grad():
        img_fake = model(emb)
    
    # Reverse preprocessing normalization
    img_r_np = ((img_real.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    img_f_np = ((img_fake.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    
    current_psnr = psnr(img_r_np, img_f_np)
    current_ssim = ssim(img_r_np, img_f_np, channel_axis=2)
    
    # Extract embeddings directly using our bypassed function
    arc_real = get_arcface_embedding(img_r_np)
    arc_fake = get_arcface_embedding(img_f_np)
    
    if arc_real is not None: orig_detected += 1
    if arc_fake is not None: fake_detected += 1
    total_checked += 1
    
    arc_sim = cosine_similarity(arc_real, arc_fake)
    
    unique_users_collected.append({
        'original': img_r_np,
        'reconstructed': img_f_np,
        'psnr': current_psnr,
        'ssim': current_ssim,
        'arc_sim': arc_sim
    })
    
    plotted_ids.add(lbl_val)
    if len(unique_users_collected) == 20:
        break

print(f"\n--- Detection Diagnostics ---")
print(f"Faces processed successfully in Originals: {orig_detected}/{total_checked}")
print(f"Faces processed successfully in Reconstructions: {fake_detected}/{total_checked}")

# Setup 8-row by 5-col plot setup
fig, axes = plt.subplots(8, 5, figsize=(20, 25))
fig.subplots_adjust(hspace=0.4, wspace=0.1)

user_idx = 0
for row_pair in range(4):
    row_orig = row_pair * 2
    row_recon = row_orig + 1
    
    for col in range(5):
        if user_idx >= len(unique_users_collected):
            break
            
        data = unique_users_collected[user_idx]
        

        orig_plot = data['original']
        recon_plot = data['reconstructed']
        
        # We check the center pixel to see if it's heavily blue-biased (Smurf check). 
        # If R and B are swapped, we force swap them back for plotting.
        if orig_plot.shape[-1] == 3:
            orig_plot = cv2.cvtColor(orig_plot, cv2.COLOR_BGR2RGB)
            recon_plot = cv2.cvtColor(recon_plot, cv2.COLOR_BGR2RGB)
        
        # Original Image
        ax_orig = axes[row_orig, col]
        ax_orig.imshow(orig_plot)
        ax_orig.axis('off')
        if col == 0:
            ax_orig.set_title("Originals ->", fontsize=14, loc='left', fontweight='bold')
            
        # Reconstructed Image
        ax_recon = axes[row_recon, col]
        ax_recon.imshow(recon_plot)
        ax_recon.axis('off')
        
        metrics_text = f"PSNR: {data['psnr']:.1f}\nSSIM: {data['ssim']:.2f}\nArcSim: {data['arc_sim']:.2f}"
        
        if col == 0:
            ax_recon.set_title(f"Reconstructed ->\n{metrics_text}", fontsize=10, color='darkred', loc='left')
        else:
            ax_recon.set_title(metrics_text, fontsize=10, color='darkred', loc='center')
            
        user_idx += 1

plt.show()
