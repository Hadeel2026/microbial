# -*- coding: utf-8 -*-
"""AI in microbial.ipynb
# Dataset Loading & Preprocessing
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import tifffile as tiff
import albumentations as A
import cv2
from glob import glob

# Dataset paths (corrected)

segmentation_path = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset"
superres_path = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Super-resolution_prediction_S.aureus"

# Brightfield patches
brightfield_train = os.path.join(segmentation_path, "brightfield_dataset/train/patches/brightfield")
brightfield_masks = os.path.join(segmentation_path, "brightfield_dataset/train/patches/masks")

# Fluorescence patches
fluorescence_train = os.path.join(segmentation_path, "fluorescence_dataset/train/patches/fluorescence")
fluorescence_masks = os.path.join(segmentation_path, "fluorescence_dataset/train/patches/masks")

# Super-resolution dataset
train_wf = os.path.join(superres_path, "train/WF")
train_sim = os.path.join(superres_path, "train/SIM")
test_wf = os.path.join(superres_path, "test/WF")
test_sim = os.path.join(superres_path, "test/SIM")

# -------------------------
# Utilities
# -------------------------
def normalize_image(img, method="zscore"):
    img = img.astype(np.float32)
    if method == "zscore":
        mean, std = img.mean(), img.std()
        return (img - mean) / (std + 1e-8)
    elif method == "minmax":
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def extract_patches(img, patch_size=256, stride=128):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.3, alpha=1, sigma=50)
])

def augment_patch(patch):
    patch = (patch * 255).astype(np.uint8)
    augmented = augment(image=patch)["image"]
    return augmented.astype(np.float32) / 255.0

# -------------------------
# Loaders
# -------------------------
def load_superres_pairs(wf_folder, sim_folder, limit=None):
    wf_files = sorted(glob(os.path.join(wf_folder, "*.tif")))
    sim_files = sorted(glob(os.path.join(sim_folder, "*.tif")))

    print(f"Found {len(wf_files)} WF and {len(sim_files)} SIM files")

    pairs = []
    for wf_file, sim_file in zip(wf_files, sim_files):
        wf = normalize_image(tiff.imread(wf_file), method="zscore")
        sim = normalize_image(tiff.imread(sim_file), method="zscore")

        if wf.shape != sim.shape:
            sim = cv2.resize(sim, (wf.shape[1], wf.shape[0]))

        wf_patches = extract_patches(wf, patch_size=256, stride=128)
        sim_patches = extract_patches(sim, patch_size=256, stride=128)

        for w, s in zip(wf_patches, sim_patches):
            pairs.append((w, s))

    if limit:
        pairs = pairs[:limit]
    return pairs

def load_segmentation_dataset(img_folder, mask_folder, limit=None):
    img_files = sorted(glob(os.path.join(img_folder, "*.tif")))
    mask_files = sorted(glob(os.path.join(mask_folder, "*.tif")))

    print(f"Found {len(img_files)} images and {len(mask_files)} masks in {img_folder}")

    data = []
    for img_f, mask_f in zip(img_files, mask_files):
        img = normalize_image(tiff.imread(img_f), method="zscore")
        mask = tiff.imread(mask_f).astype(np.uint8)

        img_patches = extract_patches(img, patch_size=256, stride=128)
        mask_patches = extract_patches(mask, patch_size=256, stride=128)

        for i, m in zip(img_patches, mask_patches):
            data.append((i, m))

    if limit:
        data = data[:limit]
    return data

# -------------------------
# Run Phase 1
# -------------------------
train_pairs = load_superres_pairs(train_wf, train_sim, limit=50)
print(" Super-resolution training pairs:", len(train_pairs))

seg_data = load_segmentation_dataset(brightfield_train, brightfield_masks, limit=50)
print("Segmentation dataset patches:", len(seg_data))

if len(seg_data) > 0:
    example_img, example_mask = seg_data[0]
    aug_img = augment_patch(example_img)
    print("Example patch shape:", example_img.shape, "Mask shape:", example_mask.shape)
else:
    print(" No segmentation patches found.")

import os
import numpy as np
import tifffile as tiff
import albumentations as A
import cv2
from glob import glob

# -------------------------
# Dataset paths
# -------------------------
segmentation_path = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset"
superres_path = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Super-resolution_prediction_S.aureus"

# Brightfield patches
brightfield_train = os.path.join(segmentation_path, "brightfield_dataset/train/patches/brightfield")
brightfield_masks = os.path.join(segmentation_path, "brightfield_dataset/train/patches/masks")

# Super-resolution dataset
train_wf = os.path.join(superres_path, "train/WF")
train_sim = os.path.join(superres_path, "train/SIM")
test_wf = os.path.join(superres_path, "test/WF")
test_sim = os.path.join(superres_path, "test/SIM")

# -------------------------
# Utilities
# -------------------------
def normalize_image(img, method="zscore"):
    img = img.astype(np.float32)
    if method == "zscore":
        mean, std = img.mean(), img.std()
        return (img - mean) / (std + 1e-8)
    elif method == "minmax":
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def extract_patches(img, patch_size=256, stride=128):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.3, alpha=1, sigma=50)
])

def augment_patch(patch):
    patch = (patch * 255).astype(np.uint8)
    augmented = augment(image=patch)["image"]
    return augmented.astype(np.float32) / 255.0

# -------------------------
# Loaders
# -------------------------
def load_superres_pairs(wf_folder, sim_folder, limit=None):
    wf_files = sorted(glob(os.path.join(wf_folder, "*.tif")))
    sim_files = sorted(glob(os.path.join(sim_folder, "*.tif")))

    print(f"Found {len(wf_files)} WF and {len(sim_files)} SIM files")

    pairs = []
    for wf_file, sim_file in zip(wf_files, sim_files):
        wf = normalize_image(tiff.imread(wf_file), method="zscore")
        sim = normalize_image(tiff.imread(sim_file), method="zscore")

        if wf.shape != sim.shape:
            sim = cv2.resize(sim, (wf.shape[1], wf.shape[0]))

        wf_patches = extract_patches(wf, patch_size=256, stride=128)
        sim_patches = extract_patches(sim, patch_size=256, stride=128)

        for w, s in zip(wf_patches, sim_patches):
            pairs.append((w, s))

    if limit:
        pairs = pairs[:limit]

    return pairs, wf_files, sim_files

def load_segmentation_dataset(img_folder, mask_folder, limit=None):
    img_files = sorted(glob(os.path.join(img_folder, "*.tif")))
    mask_files = sorted(glob(os.path.join(mask_folder, "*.tif")))

    print(f"Found {len(img_files)} images and {len(mask_files)} masks in {img_folder}")

    data = []
    for img_f, mask_f in zip(img_files, mask_files):
        img = normalize_image(tiff.imread(img_f), method="zscore")
        mask = tiff.imread(mask_f).astype(np.uint8)

        img_patches = extract_patches(img, patch_size=256, stride=128)
        mask_patches = extract_patches(mask, patch_size=256, stride=128)

        for i, m in zip(img_patches, mask_patches):
            data.append((i, m))

    if limit:
        data = data[:limit]

    return data, img_files, mask_files

# -------------------------
# Run Phase 1
# -------------------------
train_pairs, wf_files, sim_files = load_superres_pairs(train_wf, train_sim, limit=50)
seg_data, img_files, mask_files = load_segmentation_dataset(brightfield_train, brightfield_masks, limit=50)

# Example sample
if len(seg_data) > 0:
    example_img, example_mask = seg_data[0]
    aug_img = augment_patch(example_img)

# -------------------------
# Phase 1 Summary Report
# -------------------------
print("\n Dataset Summary (Phase 1)")
print("- Super-resolution:")
print(f"  • WF images: {len(wf_files)}")
print(f"  • SIM images: {len(sim_files)}")
print(f"  • Training pairs extracted: {len(train_pairs)}")

if len(train_pairs) > 0:
    wf_sample, sim_sample = train_pairs[0]
    print(f"  • Example WF patch stats: min={wf_sample.min():.2f}, max={wf_sample.max():.2f}, mean={wf_sample.mean():.2f}")
    print(f"  • Example SIM patch stats: min={sim_sample.min():.2f}, max={sim_sample.max():.2f}, mean={sim_sample.mean():.2f}")

print("\n- Segmentation:")
print(f"  • Brightfield images: {len(img_files)}")
print(f"  • Masks: {len(mask_files)}")
print(f"  • Patches extracted: {len(seg_data)}")

if len(seg_data) > 0:
    print(f"   Example patch shape: {example_img.shape}, mask unique values: {np.unique(example_mask)}")

"""# AI-based Super-Resolution Module"""

!pip install torch torchvision pytorch-msssim

import os
import math
import glob
import random
from tqdm import tqdm
from glob import glob

import numpy as np
import tifffile as tiff
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_ssim  # returns mean SSIM

# ============================================================
# Phase 2: AI-based Super-Resolution Module
# ============================================================

# Imports
import os
import math
import glob
import random
from tqdm import tqdm
from glob import glob

import numpy as np
import tifffile as tiff
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_ssim  # returns mean SSIM

# ----------------------------
# Hyperparameters / settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr = 1e-4
epochs = 40
patch_size = 256
stride = 128
num_workers = 4
# save_dir = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/superres_checkpoints"
save_dir = "/content/superres_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Toggle self-supervised (masked) training as an auxiliary loss
USE_NOISE2VOID = False  # set True to enable masked reconstruction loss

# Path variables (must match Phase 1 paths)
superres_path = "/content/drive/MyDrive/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Super-resolution_prediction_S.aureus"
train_wf = os.path.join(superres_path, "train/WF")
train_sim = os.path.join(superres_path, "train/SIM")
val_wf = os.path.join(superres_path, "test/WF")
val_sim = os.path.join(superres_path, "test/SIM")

# Optional: CARE checkpoint path
care_checkpoint_path = "/content/drive/MyDrive/AI in Microbial and Microscopic Analysis/Pretrained model/DeepBacs_Model_Super-resolution_prediction_S.aureus/weights_best.h5"

# ============================================================
# Utilities: PSNR and SSIM
# ============================================================
def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10((data_range ** 2) / mse)

def compute_ssim(pred, target):
    return pytorch_ssim(pred, target, data_range=pred.max().item() - pred.min().item() if pred.max().item()!=pred.min().item() else 1.0)

# ============================================================
# Dataset loader: WF ↔ SIM paired patches
# ============================================================
class SuperResPairDataset(Dataset):
    def __init__(self, wf_dir, sim_dir, patch_size=256, stride=128, transforms=None):
        super().__init__()
        self.wf_files = sorted(glob(os.path.join(wf_dir, "*.tif")))
        self.sim_files = sorted(glob(os.path.join(sim_dir, "*.tif")))
        assert len(self.wf_files) == len(self.sim_files), "WF and SIM counts differ"
        self.pair_files = list(zip(self.wf_files, self.sim_files))
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms

        # Build patch index
        self.index = []
        for fi, (wf_f, sim_f) in enumerate(self.pair_files):
            wf_img = tiff.imread(wf_f).astype(np.float32)
            h, w = wf_img.shape[:2]
            ys = list(range(0, max(1, h - patch_size + 1), stride))
            xs = list(range(0, max(1, w - patch_size + 1), stride))
            if len(ys) == 0: ys = [max(0, (h - patch_size) // 2)]
            if len(xs) == 0: xs = [max(0, (w - patch_size) // 2)]
            for y in ys:
                for x in xs:
                    self.index.append((fi, y, x))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, y, x = self.index[idx]
        wf_path, sim_path = self.pair_files[fi]
        wf = tiff.imread(wf_path).astype(np.float32)
        sim = tiff.imread(sim_path).astype(np.float32)

        # Align SIM size to WF
        if wf.shape != sim.shape:
            sim = cv2.resize(sim, (wf.shape[1], wf.shape[0]))

        # Normalize (z-score)
        wf = (wf - wf.mean()) / (wf.std() + 1e-8)
        sim = (sim - sim.mean()) / (sim.std() + 1e-8)

        # Convert to grayscale if needed
        if wf.ndim == 3: wf = wf[...,0]
        if sim.ndim == 3: sim = sim[...,0]

        # Extract patch safely
        h, w = wf.shape
        y = min(max(0, y), max(0, h - self.patch_size))
        x = min(max(0, x), max(0, w - self.patch_size))
        wf_patch = wf[y:y+self.patch_size, x:x+self.patch_size]
        sim_patch = sim[y:y+self.patch_size, x:x+self.patch_size]

        # Shape -> [C,H,W]
        wf_patch = np.expand_dims(wf_patch, 0).astype(np.float32)
        sim_patch = np.expand_dims(sim_patch, 0).astype(np.float32)

        wf_t = torch.from_numpy(wf_patch)
        sim_t = torch.from_numpy(sim_patch)

        if USE_NOISE2VOID:
            mask = torch.zeros_like(wf_t)
            bsize = self.patch_size // 4
            ry = random.randint(0, self.patch_size - bsize)
            rx = random.randint(0, self.patch_size - bsize)
            mask[:, ry:ry+bsize, rx:rx+bsize] = 1.0
            wf_masked = wf_t.clone()
            wf_masked[:, ry:ry+bsize, rx:rx+bsize] = 0.0
            return wf_masked, sim_t, mask
        else:
            return wf_t, sim_t

# ============================================================
# Model: UNet + Transformer bottleneck
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=4, num_layers=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dim_feedforward=channels*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H*W).permute(0,2,1)  # [B, N, C]
        tokens = self.layer_norm(tokens)
        tokens = self.transformer(tokens)
        out = tokens.permute(0,2,1).view(B, C, H, W)
        return out

class UNetTransformer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bottleneck_conv = ConvBlock(base_ch*8, base_ch*8)
        self.transformer = TransformerBottleneck(base_ch*8)
        self.up3 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck_conv(x4)
        b = self.transformer(b)
        u3 = self.up3(b, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        out = self.outc(u1)
        return out

# ============================================================
# Perceptual loss (VGG)
# ============================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.vgg_slices = nn.Sequential(*list(vgg.children())[:16]).to(device)
        for p in self.vgg_slices.parameters(): p.requires_grad = False
        self.transform = T.Normalize(mean=[0.485], std=[0.229])

    def forward(self, pred, target):
        pred3 = pred.repeat(1,3,1,1)
        target3 = target.repeat(1,3,1,1)
        pred3 = (pred3 - pred3.min()) / (pred3.max() - pred3.min() + 1e-8)
        target3 = (target3 - target3.min()) / (target3.max() - target3.min() + 1e-8)
        pred3 = self.transform(pred3)
        target3 = self.transform(target3)
        f_pred = self.vgg_slices(pred3)
        f_tgt = self.vgg_slices(target3)
        return F.l1_loss(f_pred, f_tgt)

# ============================================================
# Training and validation
# ============================================================
def train_one_epoch(model, loader, optimizer, epoch, device, perceptual_loss_fn):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        if USE_NOISE2VOID:
            wf_masked, sim, mask = batch
            wf_masked, sim, mask = wf_masked.to(device), sim.to(device), mask.to(device)
            input_tensor = wf_masked
        else:
            wf, sim = batch
            input_tensor, sim = wf.to(device), sim.to(device)

        optimizer.zero_grad()
        out = model(input_tensor)

        # supervised losses
        l1 = F.l1_loss(out, sim)
        ssim_val = pytorch_ssim(out, sim, data_range=out.max().item()-out.min().item() if out.max().item()!=out.min().item() else 1.0)
        loss_ssim = 1.0 - ssim_val
        perceptual = perceptual_loss_fn(out, sim)

        loss = l1 + 0.8 * loss_ssim + 0.1 * perceptual

        if USE_NOISE2VOID:
            mask_bool = (mask > 0.5).float()
            nv_loss = F.l1_loss(out * mask_bool, sim * mask_bool)
            loss += 0.5 * nv_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {running_loss/(i+1):.4f}")

    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    avg_psnr, avg_ssim, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for wf, sim in loader:
            wf, sim = wf.to(device), sim.to(device)
            out = model(wf)
            ps = psnr(out, sim)
            ss = pytorch_ssim(out, sim, data_range=out.max().item()-out.min().item() if out.max().item()!=out.min().item() else 1.0).item()
            avg_psnr += ps
            avg_ssim += ss
            cnt += 1
    return avg_psnr/max(1,cnt), avg_ssim/max(1,cnt)

# ============================================================
# Prepare data loaders
# ============================================================
train_dataset = SuperResPairDataset(train_wf, train_sim, patch_size=patch_size, stride=stride)
val_dataset = SuperResPairDataset(val_wf, val_sim, patch_size=patch_size, stride=stride)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")

# ============================================================
# Build model and optimizer
# ============================================================
model = UNetTransformer(in_ch=1, out_ch=1, base_ch=32).to(device)
perceptual_loss_fn = VGGPerceptualLoss(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Try loading CARE checkpoint
if os.path.exists(care_checkpoint_path):
    try:
        ck = torch.load(care_checkpoint_path, map_location=device)
        if 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'], strict=False)
        else:
            model.load_state_dict(ck, strict=False)
        print("Loaded CARE checkpoint for benchmarking.")
    except Exception as e:
        print("Could not load CARE checkpoint:", e)

# ============================================================
# Training loop
# ============================================================
best_val_ssim = -1.0
for epoch in range(1, epochs+1):
    train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device, perceptual_loss_fn)
    val_psnr, val_ssim = validate(model, val_loader, device)
    print(f"Epoch {epoch} -> Train loss {train_loss:.4f} | Val PSNR {val_psnr:.2f} | Val SSIM {val_ssim:.4f}")

    scheduler.step(val_ssim)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_psnr': val_psnr,
        'val_ssim': val_ssim
    }
    torch.save(ckpt, os.path.join(save_dir, f"superres_epoch_{epoch}.pth"))

    if val_ssim > best_val_ssim:
        best_val_ssim = val_ssim
        torch.save(ckpt, os.path.join(save_dir, "superres_best.pth"))
        print(f"New best model (SSIM={val_ssim:.4f}) saved.")

print("Training finished. Best Val SSIM:", best_val_ssim)

from torch.utils.data import Dataset, DataLoader
patch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr = 1e-4
epochs = 40
patch_size = 256
stride = 128
num_workers = 4
superres_path = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Super-resolution_prediction_S.aureus"
train_wf = os.path.join(superres_path, "train/WF")
train_sim = os.path.join(superres_path, "train/SIM")
val_wf = os.path.join(superres_path, "test/WF")
val_sim = os.path.join(superres_path, "test/SIM")
USE_NOISE2VOID = False
# ============================================================
# Dataset loader: WF ↔ SIM paired patches
# ============================================================
class SuperResPairDataset(Dataset):
    def __init__(self, wf_dir, sim_dir, patch_size=256, stride=128, transforms=None):
        super().__init__()
        self.wf_files = sorted(glob(os.path.join(wf_dir, "*.tif")))
        self.sim_files = sorted(glob(os.path.join(sim_dir, "*.tif")))
        assert len(self.wf_files) == len(self.sim_files), "WF and SIM counts differ"
        self.pair_files = list(zip(self.wf_files, self.sim_files))
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms

        # Build patch index
        self.index = []
        for fi, (wf_f, sim_f) in enumerate(self.pair_files):
            wf_img = tiff.imread(wf_f).astype(np.float32)
            h, w = wf_img.shape[:2]
            ys = list(range(0, max(1, h - patch_size + 1), stride))
            xs = list(range(0, max(1, w - patch_size + 1), stride))
            if len(ys) == 0: ys = [max(0, (h - patch_size) // 2)]
            if len(xs) == 0: xs = [max(0, (w - patch_size) // 2)]
            for y in ys:
                for x in xs:
                    self.index.append((fi, y, x))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, y, x = self.index[idx]
        wf_path, sim_path = self.pair_files[fi]
        wf = tiff.imread(wf_path).astype(np.float32)
        sim = tiff.imread(sim_path).astype(np.float32)

        # Align SIM size to WF
        if wf.shape != sim.shape:
            sim = cv2.resize(sim, (wf.shape[1], wf.shape[0]))

        # Normalize (z-score)
        wf = (wf - wf.mean()) / (wf.std() + 1e-8)
        sim = (sim - sim.mean()) / (sim.std() + 1e-8)

        # Convert to grayscale if needed
        if wf.ndim == 3: wf = wf[...,0]
        if sim.ndim == 3: sim = sim[...,0]

        # Extract patch safely
        h, w = wf.shape
        y = min(max(0, y), max(0, h - self.patch_size))
        x = min(max(0, x), max(0, w - self.patch_size))
        wf_patch = wf[y:y+self.patch_size, x:x+self.patch_size]
        sim_patch = sim[y:y+self.patch_size, x:x+self.patch_size]

        # Shape -> [C,H,W]
        wf_patch = np.expand_dims(wf_patch, 0).astype(np.float32)
        sim_patch = np.expand_dims(sim_patch, 0).astype(np.float32)

        wf_t = torch.from_numpy(wf_patch)
        sim_t = torch.from_numpy(sim_patch)

        if USE_NOISE2VOID:
            mask = torch.zeros_like(wf_t)
            bsize = self.patch_size // 4
            ry = random.randint(0, self.patch_size - bsize)
            rx = random.randint(0, self.patch_size - bsize)
            mask[:, ry:ry+bsize, rx:rx+bsize] = 1.0
            wf_masked = wf_t.clone()
            wf_masked[:, ry:ry+bsize, rx:rx+bsize] = 0.0
            return wf_masked, sim_t, mask
        else:
            return wf_t, sim_t
# ============================================================
# Prepare data loaders
# ============================================================
train_dataset = SuperResPairDataset(train_wf, train_sim, patch_size=patch_size, stride=stride)
val_dataset = SuperResPairDataset(val_wf, val_sim, patch_size=patch_size, stride=stride)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")

# ============================================================
# Model: UNet + Transformer bottleneck
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=4, num_layers=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dim_feedforward=channels*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H*W).permute(0,2,1)  # [B, N, C]
        tokens = self.layer_norm(tokens)
        tokens = self.transformer(tokens)
        out = tokens.permute(0,2,1).view(B, C, H, W)
        return out

class UNetTransformer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bottleneck_conv = ConvBlock(base_ch*8, base_ch*8)
        self.transformer = TransformerBottleneck(base_ch*8)
        self.up3 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck_conv(x4)
        b = self.transformer(b)
        u3 = self.up3(b, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        out = self.outc(u1)
        return out

# ============================================================
# Extended Phase 2: CARE Baseline vs. Transformer UNet
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as pytorch_ssim
import tifffile as tiff

import tensorflow as tf
from keras.layers import TFSMLayer
from prettytable import PrettyTable

# ----------------------------
# Paths
# ----------------------------
care_savedmodel_dir = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Pretrained model/DeepBacs_Model_Super-resolution_prediction_S.aureus/TF_SavedModel"
pytorch_best_ckpt = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/superres_checkpoints/superres_best.pth"

# ----------------------------
# Load CARE baseline (TensorFlow/Keras via TFSMLayer)
# ----------------------------
care_model = TFSMLayer(care_savedmodel_dir, call_endpoint="serving_default")
print("Loaded CARE baseline model via TFSMLayer.")

def run_inference_care(model, wf_np):
    wf_in = np.expand_dims(wf_np, (0,-1)).astype(np.float32)  # [1,H,W,1]
    pred = model(wf_in)

    # If model returns dict of outputs, take the first tensor
    if isinstance(pred, dict):
        first_key = list(pred.keys())[0]
        pred = pred[first_key]

    pred = pred.numpy()

    # Remove batch and channel dims
    pred = np.squeeze(pred)  # shape could be (H,W) or (H,W,2)

    # If CARE outputs 2 channels, just take first channel
    if pred.ndim == 3 and pred.shape[-1] > 1:
        pred = pred[..., 0]  # keep channel 0 (or np.mean(pred, axis=-1))

    return pred


# ----------------------------
# Reload validation dataset (from Phase 2)
# ----------------------------
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ----------------------------
# Reload PyTorch model
# ----------------------------
model = UNetTransformer(in_ch=1, out_ch=1, base_ch=32).to(device)
ckpt = torch.load(pytorch_best_ckpt, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Loaded PyTorch best model for inference.")

# ============================================================
# Utility Functions
# ============================================================
def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)

def compute_ssim(pred, target):
    return pytorch_ssim(pred, target, data_range=1.0).item()

def run_inference_pytorch(model, wf_torch):
    """Run PyTorch model inference"""
    with torch.no_grad():
        out = model(wf_torch.unsqueeze(0).to(device))  # [1,1,H,W]
    return out.squeeze(0).squeeze(0).cpu().numpy()

# ============================================================
# Evaluation Loop
# ============================================================
results = {"CARE": {"PSNR": [], "SSIM": []}, "Ours": {"PSNR": [], "SSIM": []}}
sample_images = []  # store examples for visualization

for i, (wf, sim) in enumerate(val_loader):
    wf, sim = wf.to(device), sim.to(device)

    # Ground truth
    sim_np = sim.squeeze(0).squeeze(0).cpu().numpy()
    wf_np = wf.squeeze(0).squeeze(0).cpu().numpy()

    # CARE output
    care_out = run_inference_care(care_model, wf_np)
    care_out_torch = torch.tensor(care_out).unsqueeze(0).unsqueeze(0).to(device)

    # Ours output
    ours_out = run_inference_pytorch(model, wf.squeeze(0))
    ours_out_torch = torch.tensor(ours_out).unsqueeze(0).unsqueeze(0).to(device)

    sim_torch = sim

    # Metrics
    results["CARE"]["PSNR"].append(psnr(care_out_torch, sim_torch))
    results["CARE"]["SSIM"].append(compute_ssim(care_out_torch, sim_torch))
    results["Ours"]["PSNR"].append(psnr(ours_out_torch, sim_torch))
    results["Ours"]["SSIM"].append(compute_ssim(ours_out_torch, sim_torch))

    # Save visualization samples
    if i < 3:
        sample_images.append((wf_np, sim_np, care_out, ours_out))

# ============================================================
# Extended Phase 2 (Finalized): CARE Baseline vs Transformer-UNet
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ============================================================
# Quantitative Results with Mean ± Std
# ============================================================
table = PrettyTable()
table.field_names = ["Model", "PSNR (mean ± std)", "SSIM (mean ± std)"]

for model_name in results:
    psnr_vals = np.array(results[model_name]["PSNR"])
    ssim_vals = np.array(results[model_name]["SSIM"])

    mean_psnr, std_psnr = np.mean(psnr_vals), np.std(psnr_vals)
    mean_ssim, std_ssim = np.mean(ssim_vals), np.std(ssim_vals)

    table.add_row([
        model_name,
        f"{mean_psnr:.2f} ± {std_psnr:.2f}",
        f"{mean_ssim:.4f} ± {std_ssim:.4f}"
    ])

print(" Final Comparison with Variability")
print(table)

# ============================================================
# Qualitative Visualization
# ============================================================
for idx, (wf_np, sim_np, care_out, ours_out) in enumerate(sample_images):
    plt.figure(figsize=(14,4))

    plt.subplot(1,4,1)
    plt.imshow(wf_np, cmap="gray")
    plt.title("WF Input")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(sim_np, cmap="gray")
    plt.title("SIM Ground Truth")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(care_out, cmap="gray")
    plt.title("CARE Baseline")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(ours_out, cmap="gray")
    plt.title("Our Model")
    plt.axis("off")

    plt.suptitle(f"Qualitative Comparison - Validation Sample {idx}", fontsize=14)
    plt.show()

# ============================================================
# Extended Phase 2 (Finalized): CARE vs. Transformer UNet
# + Training augmentations
# + Full-image reconstruction
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim as pytorch_ssim
from torchvision import transforms
import tifffile as tiff

import tensorflow as tf
from keras.layers import TFSMLayer
from prettytable import PrettyTable

# ----------------------------
# Paths
# ----------------------------
care_savedmodel_dir = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Pretrained model/DeepBacs_Model_Super-resolution_prediction_S.aureus/TF_SavedModel"
pytorch_best_ckpt = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/superres_checkpoints/superres_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dataset with Augmentations
# ============================================================
class AugmentedSuperResDataset(Dataset):
    def __init__(self, wf_files, sim_files, patch_size=256, augment=True):
        self.wf_files = wf_files
        self.sim_files = sim_files
        self.patch_size = patch_size
        self.augment = augment

        # Augmentation transforms
        self.transforms = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation([90, 90]),
            transforms.RandomRotation([180, 180]),
            transforms.RandomRotation([270, 270]),
        ])

    def __len__(self):
        return len(self.wf_files)

    def __getitem__(self, idx):
        wf = tiff.imread(self.wf_files[idx]).astype(np.float32) / 255.0
        sim = tiff.imread(self.sim_files[idx]).astype(np.float32) / 255.0

        # Crop patch if image is larger than patch_size
        H, W = wf.shape
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            wf = wf[top:top+self.patch_size, left:left+self.patch_size]
            sim = sim[top:top+self.patch_size, left:left+self.patch_size]

        wf = torch.tensor(wf).unsqueeze(0)  # [1,H,W]
        sim = torch.tensor(sim).unsqueeze(0)

        # Apply augmentations jointly
        if self.augment:
            stacked = torch.cat([wf, sim], dim=0)  # [2,H,W]
            stacked = self.transforms(stacked)
            wf, sim = stacked[0].unsqueeze(0), stacked[1].unsqueeze(0)

        return wf, sim

# ============================================================
# CARE Model Loader
# ============================================================
care_model = TFSMLayer(care_savedmodel_dir, call_endpoint="serving_default")
print(" Loaded CARE baseline model via TFSMLayer.")

def run_inference_care(model, wf_np):
    wf_in = np.expand_dims(wf_np, (0,-1)).astype(np.float32)  # [1,H,W,1]
    pred = model(wf_in)

    if isinstance(pred, dict):  # CARE outputs dict
        pred = list(pred.values())[0]  # take first tensor

    pred = pred.numpy()  # e.g. (1,256,256,2) or (256,256,2)

    # Force to first channel if multi-channel
    if pred.ndim == 4:  # (1,H,W,C)
        pred = pred[0, ..., 0]
    elif pred.ndim == 3 and pred.shape[-1] > 1:  # (H,W,C)
        pred = pred[..., 0]
    else:
        pred = np.squeeze(pred)

    return pred.astype(np.float32)  # always (H,W)



# ============================================================
# Reload PyTorch model
# ============================================================
model = UNetTransformer(in_ch=1, out_ch=1, base_ch=32).to(device)
ckpt = torch.load(pytorch_best_ckpt, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Loaded PyTorch best model for inference.")

# ============================================================
# Utility Functions
# ============================================================

def to_device(x, device=device, dtype=torch.float32):
    """Convert numpy or torch tensor to torch.FloatTensor on the right device."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=dtype)
    else:
        x = x.to(dtype)
    return x.to(device)

def psnr(pred, target, data_range=1.0):
    pred = to_device(pred)
    target = to_device(target)
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)

def compute_ssim(pred, target):
    pred = to_device(pred)
    target = to_device(target)
    return pytorch_ssim(pred, target, data_range=1.0).item()

def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)

def compute_ssim(pred, target):
    return pytorch_ssim(pred, target, data_range=1.0).item()

def run_inference_pytorch(model, wf_torch):
    with torch.no_grad():
        out = model(wf_torch.unsqueeze(0).to(device))  # [1,1,H,W]
    return out.squeeze(0).squeeze(0).cpu().numpy()

# ============================================================
# Full-Image Reconstruction from Patches
# ============================================================
def reconstruct_full_image(model, wf_np, patch_size=256, overlap=32, method="ours"):
    """Run model on overlapping patches and stitch back"""
    H, W = wf_np.shape
    stride = patch_size - overlap
    output = np.zeros((H, W))
    count = np.zeros((H, W))

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = wf_np[y:y+patch_size, x:x+patch_size]
            if method == "ours":
                patch_t = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = run_inference_pytorch(model, patch_t.squeeze(0))
            else:
                pred = run_inference_care(care_model, patch)

            output[y:y+patch_size, x:x+patch_size] += pred
            count[y:y+patch_size, x:x+patch_size] += 1

    return output / np.maximum(count, 1)

# ============================================================
# Evaluation Loop (Full Images)
# ============================================================
results = {"CARE": {"PSNR": [], "SSIM": []}, "Ours": {"PSNR": [], "SSIM": []}}
sample_images = []

for i, (wf, sim) in enumerate(val_loader):  # val_loader from Phase 2
    wf_np = wf.squeeze(0).squeeze(0).cpu().numpy()
    sim_np = sim.squeeze(0).squeeze(0).cpu().numpy()

    # Full reconstructions
    care_out = reconstruct_full_image(care_model, wf_np, method="care")
    ours_out = reconstruct_full_image(model, wf_np, method="ours")

    # Torchify for metrics
    # care_out_torch = torch.tensor(care_out).unsqueeze(0).unsqueeze(0).to(device)
    # ours_out_torch = torch.tensor(ours_out).unsqueeze(0).unsqueeze(0).to(device)
    # sim_torch = torch.tensor(sim_np).unsqueeze(0).unsqueeze(0).to(device)
    care_out_torch = to_device(care_out).unsqueeze(0).unsqueeze(0)
    ours_out_torch = to_device(ours_out).unsqueeze(0).unsqueeze(0)
    sim_torch      = to_device(sim_np).unsqueeze(0).unsqueeze(0)

    # Metrics
    results["CARE"]["PSNR"].append(psnr(care_out_torch, sim_torch))
    results["CARE"]["SSIM"].append(compute_ssim(care_out_torch, sim_torch))
    results["Ours"]["PSNR"].append(psnr(ours_out_torch, sim_torch))
    results["Ours"]["SSIM"].append(compute_ssim(ours_out_torch, sim_torch))

    # Save samples for visualization
    if i < 3:
        sample_images.append((wf_np, sim_np, care_out, ours_out))

# ============================================================
# Visualization
# ============================================================
for idx, (wf_np, sim_np, care_out, ours_out) in enumerate(sample_images):
    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1); plt.imshow(wf_np, cmap="gray"); plt.title("WF Input"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(sim_np, cmap="gray"); plt.title("SIM GT"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(care_out, cmap="gray"); plt.title("CARE Output"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(ours_out, cmap="gray"); plt.title("Ours Output"); plt.axis("off")
    plt.suptitle(f"Validation Full Image {idx}")
    plt.show()

# ============================================================
# Comparison Table
# ============================================================
table = PrettyTable()
table.field_names = ["Model", "Mean PSNR", "Std PSNR", "Mean SSIM", "Std SSIM"]

for model_name in results:
    mean_psnr = np.mean(results[model_name]["PSNR"])
    std_psnr = np.std(results[model_name]["PSNR"])
    mean_ssim = np.mean(results[model_name]["SSIM"])
    std_ssim = np.std(results[model_name]["SSIM"])
    table.add_row([model_name, f"{mean_psnr:.2f}", f"{std_psnr:.2f}",
                   f"{mean_ssim:.4f}", f"{std_ssim:.4f}"])

print(table)

"""# Segmentation Module

## wf unet, attunet & sr unet, attunet
"""

# phase3_segmentation_final_with_viz.py
"""
Phase 3 segmentation - improved full script with visualization

Saves:
 - plots/*.png => loss + metrics curves
 - viz/*.png   => sample predictions (input / GT / prob / bin / postprocessed / overlay)
 - checkpoints/*.pth => best & per-epoch & final model weights
 - phase3_results.json => experiment results
"""
import os
import json
import random
import warnings
from glob import glob
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from scipy import ndimage as ndi
from prettytable import PrettyTable

warnings.filterwarnings("ignore")

# --------- User config (edit paths) ----------
wf_img_dir = "/content/drive/MyDrive/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/brightfield"
mask_dir   = "/content/drive/MyDrive/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/masks"
sr_model_ckpt = "/content/drive/MyDrive/AI in Microbial and Microscopic Analysis/Code/superres_checkpoints/superres_best.pth"

out_dir = Path("./phase3_segmentation_results")
checkpoint_dir = out_dir / "checkpoints"
plots_dir = out_dir / "plots"
viz_dir = out_dir / "viz"
out_dir.mkdir(exist_ok=True)
checkpoint_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)
viz_dir.mkdir(exist_ok=True)

# --------- Runtime settings ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEED = 42
FAST_DEBUG = False   # Set False for full training
if FAST_DEBUG:
    EPOCHS = 4
    PATCHES_PER_IMAGE = 2
    BATCH_SIZE = 8
    EARLY_STOP = 3
else:
    EPOCHS = 50
    PATCHES_PER_IMAGE = 8
    BATCH_SIZE = 8
    EARLY_STOP = 12

LR = 1e-4
NUM_WORKERS = 0
INPUT_SIZE = 256
VAL_SPLIT = 0.2
PP_MIN_SIZE = 50  # postprocessing min blob size

EXPERIMENTS = [
    ("wf","unet"),
    ("wf","attunet"),
    ("sr","unet"),
    ("sr","attunet"),
    # ("wf+sr","unet"),
    # ("wf+sr","attunet"),
]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# --------- Utilities ----------
def safe_load_partial_state(model, ckpt_state_dict):
    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in ckpt_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped

# --------- SR model skeleton ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch,out_ch)
    def forward(self,x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX or diffY:
            x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1)
    def forward(self,x): return self.conv(x)

class SuperResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.bottleneck = DoubleConv(base*8, base*8)
        self.up1 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2)
        self.up3 = Up(base*2, base)
        self.outc = OutConv(base, out_ch)
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b  = self.bottleneck(x4)
        u = self.up1(b, x3)
        u = self.up2(u, x2)
        u = self.up3(u, x1)
        return torch.sigmoid(self.outc(u))

# Load SR model (safe partial)
sr_model = SuperResUNet(in_ch=1, out_ch=1, base=32)
sr_loaded = False
if os.path.exists(sr_model_ckpt):
    try:
        ckpt = torch.load(sr_model_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        loaded_count, skipped = safe_load_partial_state(sr_model, state)
        print(f"SR checkpoint partial load: {loaded_count} keys copied, {len(skipped)} skipped.")
        sr_loaded = loaded_count > 0
    except Exception as e:
        print("SR checkpoint load failed:", e)
else:
    print("SR checkpoint not found:", sr_model_ckpt)
sr_model.to("cpu").eval()

# --------- Losses & metrics ----------
class DiceBCE(nn.Module):
    def __init__(self, w_d=1.0, w_b=1.0, eps=1e-7):
        super().__init__()
        self.w_d = w_d; self.w_b = w_b; self.eps = eps
    def forward(self,preds,targets):
        bce = F.binary_cross_entropy(preds, targets)
        p = preds.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        dice = (2.*inter + self.eps) / (p.sum() + t.sum() + self.eps)
        return self.w_b * bce + self.w_d * (1. - dice)

def dice_coeff(pred, target, eps=1e-7):
    pred_bin = (pred >= 0.5).float()
    p = pred_bin.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return float(((2*inter + eps) / (union + eps)).mean().item())

def iou_score(pred, target, eps=1e-7):
    pred_bin = (pred >= 0.5).float()
    p = pred_bin.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return float(((inter + eps) / (union + eps)).mean().item())

def precision_recall(pred, target, eps=1e-7):
    p = (pred >= 0.5).float()
    t = (target >= 0.5).float()
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return float(prec), float(rec)

def boundary_f1(pred_np, target_np, dilation=2):
    def edges(x):
        sx = ndi.sobel(x.astype(float), axis=0)
        sy = ndi.sobel(x.astype(float), axis=1)
        return (np.hypot(sx, sy) > 0).astype(np.uint8)
    pe = edges(pred_np); te = edges(target_np)
    struct = ndi.generate_binary_structure(2,1)
    pe_d = ndi.binary_dilation(pe, structure=struct, iterations=dilation)
    te_d = ndi.binary_dilation(te, structure=struct, iterations=dilation)
    tp = np.logical_and(pe, te_d).sum()
    fp = np.logical_and(pe, ~te_d).sum()
    fn = np.logical_and(te, ~pe_d).sum()
    prec = (tp + 1e-7)/(tp + fp + 1e-7)
    rec  = (tp + 1e-7)/(tp + fn + 1e-7)
    if prec + rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

# --------- Post-processing ----------
def postprocess_mask(mask_np, min_size=PP_MIN_SIZE, open_iter=1, close_iter=1):
    m = (mask_np > 0.5).astype(np.uint8)
    if open_iter>0:
        m = ndi.binary_opening(m, iterations=open_iter)
    if close_iter>0:
        m = ndi.binary_closing(m, iterations=close_iter)
    labeled, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, labeled, range(1, n+1))
    cleaned = np.zeros_like(m)
    for i, s in enumerate(sizes, start=1):
        if s >= min_size:
            cleaned[labeled == i] = 1
    return cleaned.astype(np.uint8)

# --------- Dataset ----------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_type="wf", patch_size=256,
                 transforms=None, sr_model=None, patches_per_image=1, mode="train"):
        self.img_files = sorted(glob(os.path.join(img_dir, "*.tif")))
        self.mask_files = sorted(glob(os.path.join(mask_dir, "*.tif")))
        mask_map = {os.path.basename(m): m for m in self.mask_files}
        valid_img = []
        valid_mask = []
        for img in self.img_files:
            bn = os.path.basename(img)
            if bn in mask_map:
                valid_img.append(img); valid_mask.append(mask_map[bn])
        self.img_files = valid_img; self.mask_files = valid_mask
        assert len(self.img_files) == len(self.mask_files), "images/masks mismatch"
        self.input_type = input_type
        self.patch_size = patch_size
        self.transforms = transforms
        self.sr_model = sr_model
        self.mode = mode
        self.patches_per_image = patches_per_image if mode=="train" else 1

        self.indices = []
        for i in range(len(self.img_files)):
            for _ in range(self.patches_per_image):
                self.indices.append(i)
        print(f"Loaded {len(self.img_files)} image-mask pairs, dataset length={len(self.indices)} (mode={mode})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = self.img_files[img_idx]
        mask_path = self.mask_files[img_idx]
        wf = tiff.imread(img_path).astype(np.float32)
        mask = tiff.imread(mask_path).astype(np.uint8)

        if wf.ndim == 3:
            wf = wf[...,0]

        mn, mx = float(wf.min()), float(wf.max())
        if mx > mn:
            wf = (wf - mn) / (mx - mn)
        else:
            wf = np.zeros_like(wf, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

        h, w = wf.shape; ps = self.patch_size
        if self.mode == "train":
            top = random.randint(0, max(0, h-ps))
            left = random.randint(0, max(0, w-ps))
        else:
            top = max(0, (h-ps)//2)
            left = max(0, (w-ps)//2)

        wf_patch = wf[top:top+ps, left:left+ps]
        mask_patch = mask[top:top+ps, left:left+ps]

        if wf_patch.shape != (ps, ps):
            wf_patch = cv2.resize(wf_patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_patch, (ps, ps), interpolation=cv2.INTER_NEAREST)

        sr_out = None
        if self.input_type in ("sr","wf+sr") and self.sr_model is not None:
            with torch.no_grad():
                t = torch.from_numpy(wf_patch).unsqueeze(0).unsqueeze(0).float()
                sr_out = self.sr_model(t).cpu().numpy()[0,0]
                smn, smx = sr_out.min(), sr_out.max()
                if smx > smn:
                    sr_out = (sr_out - smn) / (smx - smn)
                else:
                    sr_out = np.zeros_like(sr_out, dtype=np.float32)

        if self.input_type == "wf":
            img = np.expand_dims(wf_patch, -1)
        elif self.input_type == "sr":
            img = np.expand_dims(sr_out if sr_out is not None else wf_patch, -1)
        elif self.input_type == "wf+sr":
            ch1 = wf_patch
            ch2 = sr_out if sr_out is not None else wf_patch
            img = np.stack([ch1, ch2], axis=-1)
        else:
            raise ValueError("input_type must be wf/sr/wf+sr")

        mask_patch = np.expand_dims(mask_patch, -1)
        if self.transforms:
            aug = self.transforms(image=img, mask=mask_patch)
            img = aug["image"]
            mask_patch = aug["mask"]

        img_t = torch.from_numpy(np.transpose(img, (2,0,1))).float()
        mask_t = torch.from_numpy(np.transpose(mask_patch, (2,0,1))).float()
        return img_t, mask_t

# --------- Segmentation models ----------
class SegDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNetSeg(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc = SegDoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*8, base*8))
        self.up1 = nn.ConvTranspose2d(base*8, base*8, 2, stride=2)
        self.conv1 = SegDoubleConv(base*16, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*4, 2, stride=2)
        self.conv2 = SegDoubleConv(base*8, base*2)
        self.up3 = nn.ConvTranspose2d(base*2, base*2, 2, stride=2)
        self.conv3 = SegDoubleConv(base*4, base)
        self.up4 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.conv4 = SegDoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.conv1(torch.cat([self.up1(x5), x4], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u1), x3], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u2), x2], dim=1))
        u4 = self.conv4(torch.cat([self.up4(u3), x1], dim=1))
        return torch.sigmoid(self.outc(u4))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g,F_int,1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l,F_int,1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(F_int,1,1), nn.Sigmoid())
    def forward(self, g, x):
        return x * self.psi(self.W_g(g) + self.W_x(x))

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc = SegDoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), SegDoubleConv(base*8, base*8))
        self.att1 = AttentionGate(base*8, base*8, base*4)
        self.att2 = AttentionGate(base*4, base*4, base*2)
        self.att3 = AttentionGate(base*2, base*2, base)
        self.att4 = AttentionGate(base, base, max(1, base//2))
        self.up1 = nn.ConvTranspose2d(base*8, base*8, 2, stride=2)
        self.conv1 = SegDoubleConv(base*16, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*4, 2, stride=2)
        self.conv2 = SegDoubleConv(base*8, base*2)
        self.up3 = nn.ConvTranspose2d(base*2, base*2, 2, stride=2)
        self.conv3 = SegDoubleConv(base*4, base)
        self.up4 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.conv4 = SegDoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d5 = self.up1(x5)
        x4_a = self.att1(d5, x4); d5 = self.conv1(torch.cat([d5, x4_a], dim=1))
        d4 = self.up2(d5)
        x3_a = self.att2(d4, x3); d4 = self.conv2(torch.cat([d4, x3_a], dim=1))
        d3 = self.up3(d4)
        x2_a = self.att3(d3, x2); d3 = self.conv3(torch.cat([d3, x2_a], dim=1))
        d2 = self.up4(d3)
        x1_a = self.att4(d2, x1); d2 = self.conv4(torch.cat([d2, x1_a], dim=1))
        return torch.sigmoid(self.outc(d2))

# --------- Training & evaluation helpers ----------
def train_one_epoch(model, loader, opt, criterion):
    model.train()
    running = []
    for imgs, masks in loader:
        imgs = imgs.to(device); masks = masks.to(device)
        opt.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        opt.step()
        running.append(loss.item())
    return float(np.mean(running)) if running else 0.0

def evaluate(model, loader, pp_min_size=PP_MIN_SIZE):
    model.eval()
    metrics = defaultdict(list)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device); masks = masks.to(device)
            preds = model(imgs)
            metrics['dice'].append(dice_coeff(preds, masks))
            metrics['iou'].append(iou_score(preds, masks))
            p, r = precision_recall(preds, masks)
            metrics['precision'].append(p); metrics['recall'].append(r)
            pb = (preds >= 0.5).float().cpu().numpy()
            mb = (masks >= 0.5).float().cpu().numpy()
            for i in range(pb.shape[0]):
                pp_mask = postprocess_mask(pb[i,0], min_size=pp_min_size)
                metrics['boundary_f1'].append(boundary_f1(pp_mask, mb[i,0]))
    summary = {}
    for k, v in metrics.items():
        arr = np.array(v) if len(v)>0 else np.array([0.0])
        summary[k] = (float(arr.mean()), float(arr.std()))
    return summary

# --------- plotting & visualization ----------
def plot_curves(train_losses, val_hist, exp_name):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='train_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(exp_name + ' Loss'); plt.grid(True); plt.legend()
    p = plots_dir / f"{exp_name}_loss.png"; plt.savefig(p, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(8,5))
    for metric in ['dice','iou','precision','recall','boundary_f1']:
        means = [m[metric][0] for m in val_hist]
        plt.plot(means, label=metric)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title(exp_name + ' Val metrics'); plt.grid(True); plt.legend()
    p = plots_dir / f"{exp_name}_metrics.png"; plt.savefig(p, bbox_inches='tight'); plt.close()

def visualize_predictions(model, dataset, n=6, save_path=None):
    model.eval()
    n = min(n, len(dataset))
    if n == 0:
        return
    idxs = np.random.choice(len(dataset), n, replace=False)

    rows = 6
    fig, axes = plt.subplots(rows, n, figsize=(n*3, rows*2.2))
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            img, mask = dataset[idx]  # img: (C,H,W), mask: (1,H,W)
            # ensure numpy arrays for plotting
            img_np = img.cpu().numpy()
            inp = img_np[0]  # first channel
            mask_np = mask.squeeze().numpy()
            # forward
            out = model(img.unsqueeze(0).to(device)).cpu().squeeze().numpy()
            if out.ndim == 3:
                out_map = out[0]
            else:
                out_map = out
            bin_pred = (out_map > 0.5).astype(np.uint8)
            pp_pred = postprocess_mask(bin_pred, min_size=PP_MIN_SIZE)
            # overlay boundaries: compute contours
            def make_overlay(img_gray, mask_bin, color=(1.0,0.0,0.0)):
                # img_gray in 0..1
                img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
                contours = cv2.findContours((mask_bin*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(img_rgb, contours, -1, (1.0,0.0,0.0), 1)
                return img_rgb
            overlay = make_overlay(inp, pp_pred)
            # Plot row by row
            axes[0,i].imshow(inp, cmap='gray'); axes[0,i].axis('off');
            axes[1,i].imshow(mask_np, cmap='gray'); axes[1,i].axis('off')
            axes[2,i].imshow(out_map, cmap='magma'); axes[2,i].axis('off')  # prob map
            axes[3,i].imshow(bin_pred, cmap='gray'); axes[3,i].axis('off')
            axes[4,i].imshow(pp_pred, cmap='gray'); axes[4,i].axis('off')
            axes[5,i].imshow(overlay); axes[5,i].axis('off')
            if i == 0:
                axes[0,i].set_title("Input")
                axes[1,i].set_title("GT")
                axes[2,i].set_title("Prob")
                axes[3,i].set_title("Bin")
                axes[4,i].set_title("Postproc")
                axes[5,i].set_title("Overlay")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# --------- run single experiment ----------
def run_single_experiment(input_type="wf", model_type="unet", epochs=EPOCHS):
    print("\n" + "="*60)
    print(f"Experiment: {input_type} + {model_type} (FAST_DEBUG={FAST_DEBUG})")

    # transforms
    train_t = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.ElasticTransform(p=0.2),
        A.Normalize(mean=0.0, std=1.0)
    ])
    val_t = A.Compose([A.Normalize(mean=0.0, std=1.0)])

    full = SegmentationDataset(wf_img_dir, mask_dir, input_type=input_type, patch_size=INPUT_SIZE,
                               transforms=None, sr_model=sr_model if sr_loaded else None,
                               patches_per_image=1, mode="train")
    N_images = len(full.img_files)
    if N_images == 0:
        raise RuntimeError("No images found - check paths")
    img_indices = np.arange(N_images)
    np.random.shuffle(img_indices)
    split = int(N_images*(1-VAL_SPLIT))
    train_imgs = img_indices[:split].tolist()
    val_imgs = img_indices[split:].tolist()

    train_ds = SegmentationDataset(wf_img_dir, mask_dir, input_type=input_type, patch_size=INPUT_SIZE,
                                   transforms=train_t, sr_model=sr_model if sr_loaded else None,
                                   patches_per_image=PATCHES_PER_IMAGE, mode="train")
    train_ds.indices = []
    for i in train_imgs:
        for _ in range(PATCHES_PER_IMAGE):
            train_ds.indices.append(i)

    val_ds = SegmentationDataset(wf_img_dir, mask_dir, input_type=input_type, patch_size=INPUT_SIZE,
                                 transforms=val_t, sr_model=sr_model if sr_loaded else None,
                                 patches_per_image=1, mode="val")
    val_ds.indices = val_imgs.copy()

    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    in_ch = 1 if input_type!="wf+sr" else 2
    base = 32
    if model_type == "unet":
        model = UNetSeg(in_ch, 1, base).to(device)
    elif model_type == "attunet":
        model = AttentionUNet(in_ch, 1, base).to(device)
    else:
        raise ValueError("model_type must be 'unet' or 'attunet'")

    print("Model params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=6)
    criterion = DiceBCE(w_d=1.0, w_b=1.0)

    best_dice = -1.0
    patience = 0
    train_losses = []
    val_history = []

    start_time = time.time()
    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, criterion)
        train_losses.append(tr_loss)
        val_metrics = evaluate(model, val_loader)
        val_history.append(val_metrics)
        scheduler.step(val_metrics['dice'][0])

        print(f"[{input_type}-{model_type}] Epoch {epoch}/{epochs} loss={tr_loss:.4f} val_dice={val_metrics['dice'][0]:.4f} (time {time.time()-t0:.1f}s)")

        # save best and export intermediate
        if val_metrics['dice'][0] > best_dice:
            best_dice = val_metrics['dice'][0]
            patience = 0
            ckpt_path = checkpoint_dir / f"{input_type}_{model_type}_best.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_metrics': val_metrics}, ckpt_path)
            torch.save(model.state_dict(), checkpoint_dir / f"{input_type}_{model_type}_epoch{epoch}.pth")
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"Early stopping after {epoch} epochs.")
                break

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes. Best dice={best_dice:.4f}")

    # load best
    ckpt_path = checkpoint_dir / f"{input_type}_{model_type}_best.pth"
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        final_metrics = ck.get('val_metrics', val_metrics)
        print("Loaded best checkpoint")
    else:
        final_metrics = val_metrics

    # save final model
    final_path = checkpoint_dir / f"{input_type}_{model_type}_final.pth"
    torch.save(model.state_dict(), final_path)

    # plots + viz
    exp_name = f"{input_type}_{model_type}"
    plot_curves(train_losses, val_history, exp_name)
    try:
        visualize_predictions(model, val_ds, n=min(6, len(val_ds)), save_path=str(viz_dir / f"{exp_name}_preds.png"))
    except Exception as e:
        print("Visualization failed:", e)

    return final_metrics

# --------- Main runner ----------
def main():
    all_results = {}
    for inp, mod in EXPERIMENTS:
        try:
            res = run_single_experiment(inp, mod, epochs=EPOCHS)
            all_results[f"{inp}_{mod}"] = res
        except Exception as e:
            print(f"Experiment {inp}_{mod} failed:", e)

    # pretty table
    table = PrettyTable()
    table.field_names = ["Exp","Dice","IoU","Prec","Rec","BF1"]
    for k, v in all_results.items():
        table.add_row([k,
                       f"{v['dice'][0]:.3f}±{v['dice'][1]:.3f}",
                       f"{v['iou'][0]:.3f}±{v['iou'][1]:.3f}",
                       f"{v['precision'][0]:.3f}±{v['precision'][1]:.3f}",
                       f"{v['recall'][0]:.3f}±{v['recall'][1]:.3f}",
                       f"{v['boundary_f1'][0]:.3f}±{v['boundary_f1'][1]:.3f}"])
    print(table)

    # save results json
    with open(out_dir / "phase3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
    
EXPERIMENTS = [
    ("wf+sr","unet"),
    ("wf+sr","attunet"),
]
def main():
    all_results = {}
    for inp, mod in EXPERIMENTS:
        try:
            res = run_single_experiment(inp, mod, epochs=EPOCHS)
            all_results[f"{inp}_{mod}"] = res
        except Exception as e:
            print(f"Experiment {inp}_{mod} failed:", e)

    # pretty table
    table = PrettyTable()
    table.field_names = ["Exp","Dice","IoU","Prec","Rec","BF1"]
    for k, v in all_results.items():
        table.add_row([k,
                       f"{v['dice'][0]:.3f}±{v['dice'][1]:.3f}",
                       f"{v['iou'][0]:.3f}±{v['iou'][1]:.3f}",
                       f"{v['precision'][0]:.3f}±{v['precision'][1]:.3f}",
                       f"{v['recall'][0]:.3f}±{v['recall'][1]:.3f}",
                       f"{v['boundary_f1'][0]:.3f}±{v['boundary_f1'][1]:.3f}"])
    print(table)

    # save results json
    with open(out_dir / "phase3_results_wf+sr.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()


# 1. Clean everything
!pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python thinc

# 2. Install numpy 1.x and a compatible OpenCV
!pip install numpy==1.26.4 opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80 opencv-contrib-python==4.9.0.80

# 3. Install StarDist, csbdeep, Cellpose
!pip install stardist==0.9.1 csbdeep==0.8.1 cellpose==4.0.6

import numpy as np
print(np.__version__)  # should print 1.26.4
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from cellpose import models as cellpose_models

import os, json, numpy as np
from tqdm import tqdm
from tabulate import tabulate
from skimage import morphology
import tifffile as tiff
from glob import glob

# Baseline imports
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from cellpose import models as cellpose_models

# ===============================
# CONFIG
# ===============================
# Input dataset dirs
wf_img_dir = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/brightfield"
mask_dir   = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/masks"

# Output dir
RESULTS_DIR = "./phase3_segmentation_results_baselines"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================
# Dataset loader (real dataset)
# ===============================
class RealDataset:
    def __init__(self, img_dir, mask_dir):
        self.img_files = sorted(glob(os.path.join(img_dir, "*.tif")))
        self.mask_files = sorted(glob(os.path.join(mask_dir, "*.tif")))
        mask_map = {os.path.basename(m): m for m in self.mask_files}
        self.images, self.masks = [], []
        for img in self.img_files:
            bn = os.path.basename(img)
            if bn in mask_map:
                self.images.append(img)
                self.masks.append(mask_map[bn])
        assert len(self.images) == len(self.masks), "Image-mask mismatch"
        print(f"Loaded {len(self.images)} image-mask pairs.")

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = tiff.imread(self.images[idx]).astype(np.float32)
        if img.ndim == 3:
            img = img[...,0]  # use first channel if RGB
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)
        mask = tiff.imread(self.masks[idx]).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        return img, mask

# ===============================
# Post-processing
# ===============================
def postprocess_mask(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask.astype(np.uint8)

# ===============================
# Metrics
# ===============================
def dice_coef(y_true, y_pred, eps=1e-7):
    inter = np.sum(y_true * y_pred)
    return (2*inter) / (np.sum(y_true)+np.sum(y_pred)+eps)

def iou_coef(y_true, y_pred, eps=1e-7):
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return inter / (union + eps)

def precision_score(y_true, y_pred, eps=1e-7):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1-y_true) * y_pred)
    return tp / (tp+fp+eps)

def recall_score(y_true, y_pred, eps=1e-7):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1-y_pred))
    return tp / (tp+fn+eps)

def boundary_f1(y_true, y_pred):
    return iou_coef(morphology.binary_dilation(y_true), morphology.binary_dilation(y_pred))

def evaluate_metrics(gt, pred):
    return dict(
        dice = dice_coef(gt, pred),
        iou = iou_coef(gt, pred),
        precision = precision_score(gt, pred),
        recall = recall_score(gt, pred),
        boundary_f1 = boundary_f1(gt, pred),
    )

# ===============================
# Baseline: StarDist
# ===============================
def run_stardist(dataset):
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    results = []
    for img, gt in tqdm(dataset, total=len(dataset), desc="StarDist"):
        img_norm = normalize(img)
        labels, _ = model.predict_instances(img_norm)
        pred = (labels > 0).astype(np.uint8)
        pred = postprocess_mask(pred)
        results.append(evaluate_metrics(gt, pred))
    return results

# ===============================
# Baseline: Cellpose
# ===============================
def run_cellpose(dataset, model_type="cyto"):
    model = cellpose_models.CellposeModel(gpu=True, pretrained_model=model_type)
    results = []
    for img, gt in tqdm(dataset, total=len(dataset), desc="Cellpose"):
        masks_pred, _, _ = model.eval([img], channels=[[0,0]])  # input wrapped in list
        pred = (masks_pred[0] > 0).astype(np.uint8)  # unwrap output
        pred = postprocess_mask(pred)
        results.append(evaluate_metrics(gt, pred))
    return results



import matplotlib.pyplot as plt
import random

def visualize_predictions(dataset, stardist_model, cellpose_model, num_samples=5, save_dir=None):
    """
    Visualize random samples with GT, StarDist, and Cellpose predictions.
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        img, gt = dataset[idx]

        # --- StarDist prediction ---
        img_norm = normalize(img)
        labels_stardist, _ = stardist_model.predict_instances(img_norm)
        pred_stardist = (labels_stardist > 0).astype(np.uint8)
        pred_stardist = postprocess_mask(pred_stardist)

        # --- Cellpose prediction ---
        masks_pred, _, _ = cellpose_model.eval([img], channels=[[0,0]])
        pred_cellpose = (masks_pred[0] > 0).astype(np.uint8)
        pred_cellpose = postprocess_mask(pred_cellpose)

        # --- Plot ---
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input Image")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_stardist, cmap="gray")
        axes[2].set_title("StarDist Prediction")
        axes[3].imshow(pred_cellpose, cmap="gray")
        axes[3].set_title("Cellpose Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"visualization_{i}.png")
            plt.savefig(out_path, dpi=150)
            print(f"Saved: {out_path}")
        plt.show()


def main():
    dataset = RealDataset(wf_img_dir, mask_dir)

    # Load models once for reuse
    stardist_model = StarDist2D.from_pretrained("2D_versatile_fluo")
    cellpose_model = cellpose_models.CellposeModel(gpu=True, pretrained_model="cyto")

    # Evaluate normally
    stardist_res = run_stardist(dataset)
    cellpose_res = run_cellpose(dataset)

    results = {
        "stardist": {k: np.mean([r[k] for r in stardist_res]) for k in stardist_res[0]},
        "cellpose": {k: np.mean([r[k] for r in cellpose_res]) for k in cellpose_res[0]},
    }

    # Save results
    with open(os.path.join(RESULTS_DIR, "phase3_baselines.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    headers = ["Exp","Dice","IoU","Prec","Rec","BF1"]
    table = []
    for exp, res in results.items():
        table.append([exp, f"{res['dice']:.3f}", f"{res['iou']:.3f}",
                           f"{res['precision']:.3f}", f"{res['recall']:.3f}", f"{res['boundary_f1']:.3f}"])
    print(tabulate(table, headers=headers, tablefmt="pretty"))

    # --- Visualization ---
    visualize_predictions(dataset, stardist_model, cellpose_model, num_samples=5,
                          save_dir=os.path.join(RESULTS_DIR, "visualizations"))


if __name__ == "__main__":
    main()

"""## Save mask pred for phase 4"""

import os
import torch
import numpy as np
import tifffile as tiff
from skimage import io
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
BASE_RESULTS = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/phase3_segmentation_results"
IMG_DIR = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Super-resolution_prediction_S.aureus/train"

# Model weight paths (best models from Phase 3)
MODELS = {
    "wf": os.path.join(BASE_RESULTS, "wf_unet", "wf_unet_best.pth"),
    "sr": os.path.join(BASE_RESULTS, "sr_unet", "sr_unet_best.pth"),
    "wf+sr": os.path.join(BASE_RESULTS, "wf+sr", "wf+sr_unet_best.pth"),
}

# ==============================
# Dataset Loader
# ==============================
class InferenceDataset(Dataset):
    def __init__(self, img_dir, exp_name="wf"):
        self.img_dir = img_dir
        self.exp_name = exp_name
        self.files = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".tif")]

        if self.exp_name == "wf+sr":
            self.wf_dir = os.path.join(img_dir, "WF")
            self.sr_dir = os.path.join(img_dir, "SIM")
            self.files = [f for f in sorted(os.listdir(self.wf_dir)) if f.endswith(".tif")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        if self.exp_name == "wf+sr":
            # Load widefield
            img_wf = io.imread(os.path.join(self.wf_dir, fname))
            if img_wf.ndim == 3:
                img_wf = img_wf[..., 0]
            img_wf = img_wf.astype(np.float32) / 255.0

            # Load super-res
            img_sr = io.imread(os.path.join(self.sr_dir, fname))
            if img_sr.ndim == 3:
                img_sr = img_sr[..., 0]
            img_sr = img_sr.astype(np.float32) / 255.0

            # Stack into (2,H,W)
            img = np.stack([img_wf, img_sr], axis=0)

        else:
            img = io.imread(os.path.join(self.img_dir, fname))
            if img.ndim == 3:
                img = img[..., 0]
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # (1,H,W)

        tensor = torch.from_numpy(img)
        return tensor, str(fname)



# ==============================
# Model Definitions (same as Phase 3!)
# ==============================
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        # Start with 32 channels (not 64!)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(256, 256))
        self.up1 = torch.nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 128)
        self.up2 = torch.nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 64)
        self.up3 = torch.nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 32)
        self.up4 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        self.outc = torch.nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        return torch.sigmoid(self.outc(x))


# ==============================
# Inference Function
# ==============================
def run_inference(exp_name, model_path, img_dir, save_root):
    print(f"[INFO] Running inference for {exp_name} using {model_path}")

    out_dir = os.path.join(save_root, exp_name, "pred_masks")
    os.makedirs(out_dir, exist_ok=True)

    dataset = InferenceDataset(img_dir, exp_name=exp_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_channels = 2 if exp_name == "wf+sr" else 1
    model = UNet(n_channels=n_channels).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        for img_tensor, fname in loader:
            img_tensor = img_tensor.to(DEVICE)

            if isinstance(fname, (list, tuple)):
                fname = fname[0]

            pred = model(img_tensor)
            pred = F.interpolate(pred, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
            pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype("uint8")

            out_path = os.path.join(out_dir, fname.replace(".tif", "_mask.tif"))
            tiff.imwrite(out_path, pred_mask)

    print(f"[INFO] Saved predictions to {out_dir}")



# MAIN
if __name__ == "__main__":
    for exp, model_path in MODELS.items():
        run_inference(exp, model_path, IMG_DIR, BASE_RESULTS)

    print("[INFO] Phase 3 inference complete. Masks are ready for Phase 4.")

"""# Downstream Biological Analysis"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tifffile as tiff
from skimage import io, measure
from scipy.stats import wasserstein_distance, mannwhitneyu, ks_2samp

# CONFIG
MASK_ROOT = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/phase3_segmentation_results"
RAW_IMG_DIR = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/brightfield"
OUT_DIR = "./phase4_bioanalysis"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

PIXEL_SIZE = 1.0

EXPERIMENTS = {
    "wf": os.path.join(MASK_ROOT, "wf", "pred_masks"),
    "sr": os.path.join(MASK_ROOT, "sr", "pred_masks"),
    "wf+sr": os.path.join(MASK_ROOT, "wf+sr", "pred_masks"),
}

# Feature Extraction

def extract_features(mask_path, raw_img_path):
    mask = tiff.imread(mask_path).astype(bool)
    if not os.path.exists(raw_img_path):
        print(f"[WARN] Image not found for mask '{os.path.basename(mask_path)}'. Skipping.")
        return []

    img = io.imread(raw_img_path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]

    img /= img.max() if img.max() > 0 else 1.0

    labeled = measure.label(mask)
    props = measure.regionprops(labeled, intensity_image=img)

    features = []
    for p in props:
        if p.area < 20:  # filter noise
            continue
        area_um = p.area * (PIXEL_SIZE ** 2)
        perimeter_um = p.perimeter * PIXEL_SIZE
        circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-6)

        features.append({
            "label": p.label,
            "area_px": p.area,
            "area_um2": area_um,
            "perimeter_um": perimeter_um,
            "circularity": circularity,
            "eccentricity": p.eccentricity,
            "solidity": p.solidity,
            "aspect_ratio": p.major_axis_length / (p.minor_axis_length + 1e-6),
            "mean_intensity": p.mean_intensity,
        })
    return features


# Analysis per experiment

def process_experiment(exp_name, mask_dir, raw_dir):
    print(f"[INFO] Extracting features for {exp_name} from {mask_dir} ...")

    features = []
    for fname in sorted(os.listdir(mask_dir)):
        if not fname.endswith("_mask.tif"):
            continue

        mask_path = os.path.join(mask_dir, fname)
        raw_fname = fname.replace("_mask", "")
        raw_path = os.path.join(raw_dir, raw_fname)

        feats = extract_features(mask_path, raw_path)
        for f in feats:
            f["image"] = raw_fname
            f["experiment"] = exp_name
            features.append(f)

    df = pd.DataFrame(features)
    out_csv = os.path.join(OUT_DIR, f"features_{exp_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved {len(df)} object-level rows for {exp_name} -> {out_csv}")
    return df


# Plots & Validation

def make_plots(df, feature, out_prefix):
    for exp in df["experiment"].unique():
        vals = df[df["experiment"] == exp][feature].dropna()
        plt.hist(vals, bins=20, alpha=0.5, label=exp)
    plt.legend()
    plt.title(f"{feature} distribution")
    plt.savefig(os.path.join(OUT_DIR, "plots", f"{out_prefix}_distribution.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column=feature, by="experiment")
    plt.title(f"{feature} per experiment")
    plt.suptitle("")
    plt.savefig(os.path.join(OUT_DIR, "plots", f"{out_prefix}_boxplot.png"))
    plt.close()

def compute_validation(df):
    results = {}
    for feature in ["area_um2", "aspect_ratio", "mean_intensity", "circularity", "eccentricity", "solidity"]:
        wf_vals = df[df["experiment"] == "wf"][feature].dropna().values
        sr_vals = df[df["experiment"] == "sr"][feature].dropna().values

        if len(wf_vals) > 0 and len(sr_vals) > 0:
            mannwhitney_p = mannwhitneyu(wf_vals, sr_vals, alternative="two-sided").pvalue
            ks_p = ks_2samp(wf_vals, sr_vals).pvalue

            results[feature] = {
                "mean_wf": float(np.mean(wf_vals)),
                "std_wf": float(np.std(wf_vals)),
                "mean_sr": float(np.mean(sr_vals)),
                "std_sr": float(np.std(sr_vals)),
                "n_wf": len(wf_vals),
                "n_sr": len(sr_vals),
                "wasserstein": float(wasserstein_distance(wf_vals, sr_vals)),
                "mannwhitney_p": float(mannwhitney_p),
                "ks_p": float(ks_p)
            }
    return results

# MAIN
if __name__ == "__main__":
    dfs = []
    for exp, mask_dir in EXPERIMENTS.items():
        if os.path.exists(mask_dir):
            df = process_experiment(exp, mask_dir, RAW_IMG_DIR)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"[WARN] No features extracted for {exp}. Check masks/raw mapping.")

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_csv = os.path.join(OUT_DIR, "all_features.csv")
        all_df.to_csv(all_csv, index=False)
        print(f"[INFO] Saved aggregated features -> {all_csv} (total objects: {len(all_df)})")

        # Plots
        for feat in ["area_um2", "aspect_ratio", "mean_intensity", "circularity", "eccentricity", "solidity"]:
            if feat in all_df.columns:
                make_plots(all_df, feat, feat)

        # Validation + Stats
        metrics = compute_validation(all_df)
        with open(os.path.join(OUT_DIR, "validation.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("[INFO] Phase 4 complete with statistical testing. Outputs in:", OUT_DIR)

    else:
        print("[ERROR] No features extracted from any experiment.")


import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Paths (adjust if needed)
WF_MASKS = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/phase3_segmentation_results/wf/pred_masks"
SR_MASKS = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Code/phase3_segmentation_results/sr/pred_masks"
IMG_DIR = "/content/drive/MyDrive/hadeel/AI in Microbial and Microscopic Analysis/Dataset/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset/train/patches/brightfield"   # raw WF images directory
SAVE_DIR = "./phase4_bioanalysis/qualitative"

os.makedirs(SAVE_DIR, exist_ok=True)

def overlay_masks(image, wf_mask, sr_mask, title, save_path):
    plt.figure(figsize=(12, 6))

    # Raw image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Raw Image")
    plt.axis("off")

    # WF mask
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap="gray")
    plt.imshow(wf_mask, cmap="Reds", alpha=0.5)
    plt.title("WF Segmentation")
    plt.axis("off")

    # SR mask
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(sr_mask, cmap="Blues", alpha=0.5)
    plt.title("SR Segmentation")
    plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def sample_and_plot(n_samples=5):
    wf_files = sorted([f for f in os.listdir(WF_MASKS) if f.endswith(".tif")])
    sr_files = sorted([f for f in os.listdir(SR_MASKS) if f.endswith(".tif")])

    # Match WF and SR masks by filename stem
    common_files = list(set(wf_files).intersection(set(sr_files)))
    if len(common_files) == 0:
        print("[WARN] No matching WF/SR masks found.")
        return

    sampled_files = random.sample(common_files, min(n_samples, len(common_files)))

    for fname in sampled_files:
        wf_mask_path = os.path.join(WF_MASKS, fname)
        sr_mask_path = os.path.join(SR_MASKS, fname)

        # Try matching raw image (remove "_mask" suffix)
        img_name = fname.replace("_mask", "")
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Raw image not found for {fname}, skipping.")
            continue

        # Load
        image = io.imread(img_path)
        wf_mask = io.imread(wf_mask_path)
        sr_mask = io.imread(sr_mask_path)

        save_path = os.path.join(SAVE_DIR, fname.replace(".tif", "_qualitative.png"))
        overlay_masks(image, wf_mask, sr_mask, title=fname, save_path=save_path)
        print(f"[INFO] Saved qualitative comparison -> {save_path}")


if __name__ == "__main__":
    sample_and_plot(n_samples=5)
    print("[INFO] Qualitative WF vs SR plots saved in:", SAVE_DIR)

