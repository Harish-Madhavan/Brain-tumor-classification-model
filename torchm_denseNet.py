# --- IMPORTS (Add SSIM) ---
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from skimage.filters import threshold_otsu
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from multiprocessing import freeze_support
import traceback # For detailed error printing
from pytorch_msssim import ssim, MS_SSIM # Import SSIM

# --- CBAM Components (No changes needed here) ---
class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att

class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_features = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_features))
        return x * spatial_att

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# --- Helper Functions ---
# ... (get_device, equalize_histogram, advanced_preprocess, to_float_tensor remain the same) ...
def get_device():
    """Helper to get the best available device (Prioritizes XPU if available)"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("Using XPU device.")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("XPU not available, using CUDA device.")
        return torch.device("cuda")
    else:
        print("XPU and CUDA not available, using CPU device.")
        return torch.device("cpu")

def equalize_histogram(img):
    """Apply histogram equalization to a grayscale PIL image."""
    return ImageOps.equalize(img)

def advanced_preprocess(img):
    """Apply CLAHE and Gaussian blur for preprocessing a grayscale PIL image."""
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0) # Keep blur moderate
    return Image.fromarray(img_blur)

def to_float_tensor(img):
    """Convert a tensor to float."""
    return img.float()

# --- Data Loading and Preprocessing (No changes needed here) ---
class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', apply_normalize=True):
        self.image_paths = []
        self.labels = []
        class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_labels_map = {folder: i for i, folder in enumerate(class_folders)}
        self.class_counts = {folder: 0 for folder in class_folders}
        self.apply_normalize = apply_normalize

        print(f"Loading {split} data from directory: {data_dir}")
        for class_folder in class_folders:
            class_dir = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            print(f"  Processing folder: {class_folder}")
            try:
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        image_path = os.path.join(class_dir, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(self.class_labels_map[class_folder])
                        self.class_counts[class_folder] += 1
            except Exception as e:
                print(f"    Error listing files in {class_dir}: {e}")

        self.image_size = image_size
        self.split = split
        # IMPORTANT: AE should generally NOT use normalization like mean/std subtraction
        # It relies on reconstructing the original pixel value distribution (often [0, 1])
        # So, use the same base transforms for AE input as for classifier input *before* normalization
        self.base_transform_list = [
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(), # Scales to [0, 1]
            transforms.Lambda(to_float_tensor),
        ]
        self.normalize_transform = None # Normalization applied externally if needed (for classifier)

        # Augmentations only for the classifier training set usually
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjusted saturation/hue
            transforms.RandomGrayscale(p=0.1),
        ])
        print(f"{split} data loading complete. Found {len(self.image_paths)} images.")
        print(f"Class Distribution for {split} set: {self.class_counts}")

    def set_normalization(self, normalize_transform):
        self.normalize_transform = normalize_transform
        print("Normalization transform set for dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            base_pipeline = transforms.Compose(self.base_transform_list)
            image = base_pipeline(image_pil) # Base transforms applied

            # Apply augmentations only during training for classifier
            if self.split == 'train' and self.apply_normalize: # Assume norm only needed for classifier
                image = self.augment_transforms(image)

            # Apply normalization only if requested (for classifier)
            if self.apply_normalize and self.normalize_transform:
                 image = self.normalize_transform(image)

            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading or processing image: {image_path}, error: {e}")
            print(traceback.format_exc())
            # Return dummy data of correct shape but recognizable label
            return torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float), torch.tensor(-1, dtype=torch.long)

class NotumorMRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train'):
        self.image_paths = []
        notumor_folder = 'notumor'
        print(f"Loading {split} 'notumor' data from directory: {data_dir}")
        notumor_dir = os.path.join(data_dir, notumor_folder)
        if not os.path.isdir(notumor_dir):
             print(f"Warning: 'notumor' directory not found: {notumor_dir}")
        else:
            print(f"  Processing folder: {notumor_folder}")
            try:
                for filename in os.listdir(notumor_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        image_path = os.path.join(notumor_dir, filename)
                        self.image_paths.append(image_path)
            except Exception as e:
                 print(f"    Error listing files in {notumor_dir}: {e}")
        self.image_size = image_size
        self.split = split
        # AE Training Transforms: Resize, Preprocess, ToTensor [0,1]
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ])
        # Simple augmentation for AE (optional, can help robustness)
        self.augment_ae = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
        ])
        print(f"{split} 'notumor' data loading complete. Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = self.transforms(image_pil) # Apply base transforms
            # Apply AE augmentations during training
            if self.split == 'train':
                 image = self.augment_ae(image)
            # Target is the non-augmented image (or augmented if using pure reconstruction)
            # Let's use the augmented image as target as well, common practice for AE
            return image, image # Target is the (potentially augmented) image itself
        except Exception as e:
            print(f"Error loading or processing 'notumor' image: {image_path}, error: {e}")
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))


# --- 2. Autoencoder Model with CBAM (Improved Skip Connections) ---

class EnhancedAutoencoder(nn.Module):
    def __init__(self, cbam_reduction_ratio=8, cbam_kernel_size=5):
        super(EnhancedAutoencoder, self).__init__()
        print(f"Initializing Enhanced Autoencoder with CBAM (reduction={cbam_reduction_ratio}, kernel={cbam_kernel_size}) - Using CBAM features for skip connections")
        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_enc1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_enc2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_enc3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)
        self.pool3 = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.cbam_bottleneck = CBAM(256, cbam_reduction_ratio, cbam_kernel_size)

        # Original attention branch
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            # Concatenation input size changed (128 + 128)
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_dec3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
             # Concatenation input size changed (64 + 64)
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_dec2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
             # Concatenation input size changed (32 + 32)
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_dec1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1) # Final output

    def forward(self, x):
        # --- Encoder (Apply CBAM before saving for skip) ---
        e1_proc = self.enc1(x)
        e1 = self.cbam_enc1(e1_proc) # Apply CBAM
        p1 = self.pool1(e1)

        e2_proc = self.enc2(p1)
        e2 = self.cbam_enc2(e2_proc) # Apply CBAM
        p2 = self.pool2(e2)

        e3_proc = self.enc3(p2)
        e3 = self.cbam_enc3(e3_proc) # Apply CBAM
        p3 = self.pool3(e3)

        # --- Bottleneck ---
        b_proc = self.bottleneck(p3)
        b = self.cbam_bottleneck(b_proc) # Apply CBAM

        # Calculate attention map from CBAM-refined bottleneck features
        att_map = self.attention(b) # Keep this explicit attention map

        # --- Decoder ---
        d3_up = self.up3(b)
        # Concatenate with CBAM-refined encoder feature (e3)
        d3_cat = torch.cat([d3_up, e3], dim=1) # <<< CHANGED
        d3_processed = self.dec3(d3_cat)
        d3 = self.cbam_dec3(d3_processed) # Apply CBAM after processing

        d2_up = self.up2(d3)
        # Concatenate with CBAM-refined encoder feature (e2)
        d2_cat = torch.cat([d2_up, e2], dim=1) # <<< CHANGED
        d2_processed = self.dec2(d2_cat)
        d2 = self.cbam_dec2(d2_processed) # Apply CBAM

        d1_up = self.up1(d2)
        # Concatenate with CBAM-refined encoder feature (e1)
        d1_cat = torch.cat([d1_up, e1], dim=1) # <<< CHANGED
        d1_processed = self.dec1(d1_cat)
        d1 = self.cbam_dec1(d1_processed) # Apply CBAM

        # --- Output ---
        # Use Sigmoid since ToTensor scales input to [0, 1] and we want output in same range
        reconstruction = torch.sigmoid(self.out_conv(d1))

        return reconstruction, att_map

# --- 3. DenseNet121 Model with CBAM for Classification (No changes needed here) ---
class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7):
        super(DenseNet121TumorClassificationModel, self).__init__()
        print(f"Initializing DenseNet121 Classifier with CBAM (reduction={cbam_reduction_ratio}, kernel={cbam_kernel_size})")
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)
        original_conv0 = self.densenet121.features.conv0
        self.densenet121.features.conv0 = nn.Conv2d(1, original_conv0.out_channels,
                                                   kernel_size=original_conv0.kernel_size, stride=original_conv0.stride,
                                                   padding=original_conv0.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                self.densenet121.features.conv0.weight.copy_(original_conv0.weight.mean(dim=1, keepdim=True))
        num_ftrs = self.densenet121.classifier.in_features
        self.cbam = CBAM(num_ftrs, reduction_ratio=cbam_reduction_ratio, kernel_size=cbam_kernel_size)
        print(f"Added CBAM block after DenseNet features with {num_ftrs} channels.")
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.densenet121.classifier = nn.Identity()
        self._initialize_weights(self.classifier_head)

    def _initialize_weights(self, classifier_module):
        for m in classifier_module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.densenet121.features(x)
        features = self.cbam(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier_head(out)
        return out

# --- 4. Utility Functions (calculate_mean_std unchanged) ---
def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_samples = 0
    num_batches = 0
    print("Calculating mean and std...")
    # Ensure loader provides images in the expected format
    for data in tqdm(loader, desc="Mean/Std Calc"):
        # Handle potential variations in loader output (e.g., SubsetRandomSampler)
        if isinstance(data, (list, tuple)) and len(data) >= 1:
            images = data[0] # Assume images are the first element
            if isinstance(images, torch.Tensor) and images.nelement() > 0:
                 # Basic sanity check for shape (assuming NCHW, C=1)
                if images.ndim == 4 and images.shape[1] == 1:
                    batch_samples = images.size(0)
                    images_flat = images.view(batch_samples, images.size(1), -1)
                    mean += images_flat.mean(2).sum(0)
                    std += images_flat.std(2).sum(0)
                    total_samples += batch_samples
                    num_batches += 1
                else:
                    # print(f"Skipping batch with unexpected shape: {images.shape}")
                    pass # Skip silently to avoid flooding console
            # else: print("Skipping non-tensor or empty image batch")
        # else: print("Skipping batch with unexpected data format")

    if total_samples == 0:
        print("Warning: No valid samples found for mean/std calculation. Using defaults [0.5, 0.5].")
        return torch.tensor([0.5]), torch.tensor([0.5]) # Return tensors
    mean /= total_samples
    std /= total_samples
    # Clamp std deviation to avoid division by zero or instability
    std = torch.max(std, torch.tensor(1e-6).to(std.device))
    return mean, std


# --- 5. Training Loops (AE Training Updated for SSIM Loss) ---

def train_enhanced_autoencoder(model, train_loader, num_epochs, device,
                               model_path='enhanced_autoencoder_notumor.pth',
                               attention_weight=0.05, ssim_weight=0.15): # Added SSIM weight
    if os.path.exists(model_path):
        print(f"Loading existing autoencoder from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Autoencoder loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading autoencoder model: {e}. Training from scratch.")

    model = model.to(device)
    criterion_mse = nn.MSELoss()
    # SSIM Loss (Note: ssim returns similarity [0,1], so loss is 1-ssim)
    # data_range=1.0 because ToTensor scales images to [0, 1]
    # size_average=True is deprecated, reduction='mean' is default
    criterion_ssim = lambda pred, target: 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5) # Increased patience

    print(f"--- Starting Enhanced Autoencoder Training ({num_epochs} epochs) ---")
    print(f"Using Loss: {(1-ssim_weight):.2f}*MSE + {ssim_weight:.2f}*SSIM + {attention_weight:.2f}*AttentionL1")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss, mse_loss_epoch, ssim_loss_epoch, att_loss_epoch = 0.0, 0.0, 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [AE Train]")

        for images, targets in progress_bar:
            # Skip potential error batches from dataloader
            if not isinstance(images, torch.Tensor) or images.shape[0] == 0: continue
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            reconstructed, attention_maps = model(images)

            # Ensure reconstruction has same shape as target for loss calculation
            if reconstructed.shape != targets.shape:
                 print(f"Shape mismatch: Recon {reconstructed.shape}, Target {targets.shape}. Skipping batch.")
                 continue

            # Calculate individual losses
            loss_mse = criterion_mse(reconstructed, targets)
            loss_ssim_val = criterion_ssim(reconstructed, targets)
            loss_att = torch.mean(torch.abs(attention_maps)) # L1 penalty on attention map

            # Combined loss
            loss = (1 - ssim_weight) * loss_mse + ssim_weight * loss_ssim_val + attention_weight * loss_att

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            mse_loss_epoch += loss_mse.item()
            ssim_loss_epoch += loss_ssim_val.item() # Store the SSIM loss value
            att_loss_epoch += loss_att.item()
            progress_bar.set_postfix(Loss=f"{loss.item():.4f}", MSE=f"{loss_mse.item():.4f}", SSIM=f"{loss_ssim_val.item():.4f}", AttL=f"{loss_att.item():.4f}")

        avg_total_loss = total_loss / len(train_loader)
        avg_mse_loss = mse_loss_epoch / len(train_loader)
        avg_ssim_loss = ssim_loss_epoch / len(train_loader)
        avg_att_loss = att_loss_epoch / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Completed: Total Loss: {avg_total_loss:.4f} (MSE: {avg_mse_loss:.4f}, SSIM: {avg_ssim_loss:.4f}, Att: {avg_att_loss:.4f})")
        scheduler.step(avg_total_loss) # Step LR scheduler based on total loss

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best AE model saved to {model_path} (Loss: {best_loss:.4f})")

    print("--- Enhanced Autoencoder training complete. ---")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best autoencoder model from {model_path}")
    except Exception as e:
         print(f"Warning: Could not reload best AE model after training: {e}")
    return model

# --- Classifier Training (train_classifier unchanged) ---
def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs=50, device=get_device(),
                    model_path='tumor_classification_densenet121.pth'):
    if os.path.exists(model_path):
        print(f"Loading existing classifier model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Classifier loaded successfully. Skipping training.")
            return model
        except Exception as e:
            print(f"Error loading classifier model: {e}. Training from scratch.")

    model.to(device)
    best_val_f1 = 0.0 # Use F1 score for saving best model

    print(f"--- Starting Classifier Training ({num_epochs} epochs) ---")
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        all_labels_train, all_preds_train = [], []
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(progress_bar_train):
            # Skip error batches (identified by label -1 in MRIDataset)
            if torch.any(labels == -1):
                # print(f"Skipping batch {batch_idx} due to loading error indicator.")
                continue
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
            optimizer.step()
            if isinstance(scheduler, OneCycleLR): # Step OneCycleLR per batch
                 scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels_train.extend(labels.cpu().numpy())
            all_preds_train.extend(predicted.cpu().numpy())
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar_train.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{current_lr:.6f}")

        # Ensure train loader wasn't empty / all batches skipped
        if not all_labels_train:
             print(f"Epoch {epoch+1}/{num_epochs}: No valid training batches processed. Skipping epoch summary.")
             continue

        avg_train_loss = train_loss / len(progress_bar_train) # Use progress bar len in case of skips
        train_accuracy = 100 * accuracy_score(all_labels_train, all_preds_train)
        train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_labels_val, all_preds_val = [], []
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in progress_bar_val:
                if torch.any(labels == -1): continue # Skip error batches
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(predicted.cpu().numpy())
                progress_bar_val.set_postfix(Loss=f"{loss.item():.4f}")

        # Ensure validation loader wasn't empty / all batches skipped
        if not all_labels_val:
             print(f"Epoch {epoch+1}/{num_epochs}: No valid validation batches processed.")
             avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0, 0.0
        else:
            avg_val_loss = val_loss / len(progress_bar_val)
            val_accuracy = 100 * accuracy_score(all_labels_val, all_preds_val)
            val_precision = precision_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
            val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")
        print(f"  Val P/R: {val_precision:.4f}/{val_recall:.4f} | LR: {current_lr:.6f}")

        # Step ReduceLROnPlateau scheduler based on validation F1
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_f1)

        # Save best model based on validation F1
        if val_f1 > best_val_f1 and len(all_labels_val) > 0: # Ensure F1 is valid
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  New best classifier model saved to {model_path} (Val F1: {best_val_f1:.4f})")

    print("--- Classifier training complete. ---")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best classifier model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not reload best classifier model after training: {e}")
    return model


# --- 6. Enhanced Tumor Outlining Function (Improved Contour Filtering) ---

def enhance_outline_detection(reconstruction_error_np, attention_map_np, threshold=None,
                                min_solidity=0.6): # Added solidity filter
    """Combine maps, filter, threshold, refine morphology, and filter contours."""
    # Normalize maps robustly
    rec_min, rec_max = reconstruction_error_np.min(), reconstruction_error_np.max()
    att_min, att_max = attention_map_np.min(), attention_map_np.max()
    # Add epsilon to prevent division by zero if an image is completely black/white
    rec_norm = (reconstruction_error_np - rec_min) / (rec_max - rec_min + 1e-8)
    att_norm = (attention_map_np - att_min) / (att_max - att_min + 1e-8)

    # Combine (Weight attention slightly higher?) - *Tunable*
    combined_map = np.clip((0.4 * rec_norm + 0.6 * att_norm), 0, 1)
    combined_uint8 = (combined_map * 255).astype(np.uint8)

    # Filtering (Consider applying after thresholding if Otsu struggles)
    combined_gauss = cv2.GaussianBlur(combined_uint8, (5, 5), 0)
    combined_median = cv2.medianBlur(combined_gauss, 5) # Median is good for salt-pepper noise
    combined_filtered_uint8 = combined_median
    combined_filtered = combined_filtered_uint8.astype(np.float32) / 255.0

    # Thresholding
    if threshold is None:
        # Check if the filtered map has any non-zero values before applying Otsu
        if combined_filtered_uint8.max() > 0:
            try:
                 otsu_thresh_val = threshold_otsu(combined_filtered_uint8)
                 threshold = otsu_thresh_val / 255.0
                 print(f"  Otsu threshold determined: {threshold:.4f} (Value: {otsu_thresh_val})")
            except Exception as e: # Catch potential errors in Otsu if image is weird
                 print(f"  Error during Otsu thresholding: {e}. Using default high threshold.")
                 threshold = 0.95 # Fallback threshold
        else:
            threshold = 0.99 # High threshold if map is all black
            print("  Warning: Filtered map is all black, using default high threshold.")
    else:
        print(f"  Using provided threshold: {threshold:.4f}")

    # Apply threshold
    tumor_mask = (combined_filtered > threshold).astype(np.uint8)

    # Morphological Refinement (Applied to binary mask)
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Closing first to fill holes within potential tumor areas
    mask_closed = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel_ellipse_5, iterations=2)
    # Opening after to remove small noise speckles introduced or remaining
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_ellipse_3, iterations=2)
    tumor_mask_refined = mask_opened

    # --- Find & Filter Contours (Added Solidity Filter) ---
    contours, hierarchy = cv2.findContours(tumor_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Contours found initially: {len(contours)}")

    valid_contours = []
    if contours:
        image_area = tumor_mask_refined.shape[0] * tumor_mask_refined.shape[1]
        min_area = 50 # Minimum pixel area
        max_area = image_area * 0.60 # Max area (e.g., 60% of image)

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                # Calculate Solidity
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity >= min_solidity:
                        valid_contours.append(c)
                    # else: print(f"    Contour rejected: Solidity {solidity:.2f} < {min_solidity}")
                # else: print(f"    Contour rejected: Hull area is 0")
            # else: print(f"    Contour rejected: Area {area} out of bounds ({min_area}-{max_area})")

        print(f"  Contours remaining after area and solidity filtering: {len(valid_contours)}")
        # Optional: Keep only the largest N valid contours
        # valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

    return tumor_mask_refined, combined_filtered, valid_contours # Return valid contours


# --- Modified Outlining Wrapper ---
def outline_tumor_enhanced(
    original_image_pil, # Pass the original PIL image for final overlay
    input_tensor_ae, # The EXACT tensor fed into the AE (Batch dim removed, CPU)
    reconstructed_tensor_ae, # The EXACT tensor output by AE (Batch dim removed, CPU)
    attention_map_tensor_ae, # The EXACT attention map from AE (Batch dim removed, CPU)
    image_size=(256, 256), # Target size
    error_threshold=None,
    min_solidity=0.6,
    show_plots=True):
    """
    Outlines tumor using Enhanced AE outputs, ensuring input consistency.
    Accepts tensors directly from AE inference. Tensors should be on CPU, no batch dim.
    """
    print(f"\n--- Starting Tumor Outlining ---")
    try:
        # --- Ensure tensors are numpy arrays on CPU ---
        if not isinstance(input_tensor_ae, np.ndarray):
             input_img_ae_np = input_tensor_ae.squeeze().cpu().numpy()
        else: input_img_ae_np = input_tensor_ae # Assume already numpy

        if not isinstance(reconstructed_tensor_ae, np.ndarray):
            reconstructed_img_np = reconstructed_tensor_ae.squeeze().cpu().numpy()
        else: reconstructed_img_np = reconstructed_tensor_ae

        if not isinstance(attention_map_tensor_ae, np.ndarray):
            attention_map_np = attention_map_tensor_ae.squeeze().cpu().numpy()
        else: attention_map_np = attention_map_tensor_ae

        # --- Resize original PIL image for overlay ---
        original_img_resized_pil = original_image_pil.resize(image_size)
        original_img_resized_np = np.array(original_img_resized_pil)

        # --- Calculate Error & Upsample Attention ---
        # Error between the *actual* AE input and its reconstruction
        reconstruction_error_np = np.abs(input_img_ae_np - reconstructed_img_np)

        # Upsample attention map using torch.nn.functional for potentially better quality
        # Add channel and batch dim for interpolate, then remove
        att_tensor_unsqueezed = torch.from_numpy(attention_map_np).unsqueeze(0).unsqueeze(0)
        attention_map_resized_tensor = F.interpolate(
            att_tensor_unsqueezed,
            size=image_size, # Target H, W
            mode='bilinear', # Or 'bicubic'
            align_corners=False
        )
        attention_map_resized_np = attention_map_resized_tensor.squeeze().cpu().numpy()


        # --- Generate Mask & Filter Contours ---
        print("Running enhanced outline detection...")
        tumor_mask_refined, combined_map_filtered, valid_contours = enhance_outline_detection(
            reconstruction_error_np, attention_map_resized_np,
            threshold=error_threshold, min_solidity=min_solidity
        )

        # --- Create Outlined Image ---
        # Start with the resized original image in BGR for coloring contours
        outlined_image_np = cv2.cvtColor(original_img_resized_np, cv2.COLOR_GRAY2BGR)
        contours_drawn = 0

        if valid_contours:
            overlay = outlined_image_np.copy()
            # Maybe limit to top N contours by area if many small ones remain
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:5] # E.g., Top 5

            for contour in valid_contours:
                # Draw filled contour on overlay (e.g., semi-transparent green)
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), cv2.FILLED)
                # Draw outline on the main image (e.g., solid red)
                cv2.drawContours(outlined_image_np, [contour], -1, (255, 0, 0), 2)
                contours_drawn += 1

            # Blend the overlay with the outlined image
            alpha = 0.3 # Transparency of the fill
            cv2.addWeighted(overlay, alpha, outlined_image_np, 1 - alpha, 0, outlined_image_np)
            print(f"Drawn {contours_drawn} valid contours.")
        else:
            print("No valid contours found to draw.")


        # --- Plotting ---
        if show_plots:
            plt.figure(figsize=(20, 5)) # Wider figure
            titles = ['Original (Resized)', 'AE Input', 'AE Recon', 'Recon Error', 'Att Map (Resized)', 'Combined Map', 'Final Mask', 'Outlined Result']
            images_to_plot = [
                original_img_resized_np, # Original
                input_img_ae_np,       # AE Input
                reconstructed_img_np,  # AE Reconstruction
                reconstruction_error_np, # Error map
                attention_map_resized_np, # Attention map
                combined_map_filtered,   # Combined heatmap before threshold
                tumor_mask_refined,    # Final binary mask
                cv2.cvtColor(outlined_image_np, cv2.COLOR_BGR2RGB) # Result
            ]
            cmaps = ['gray', 'gray', 'gray', 'hot', 'viridis', 'hot', 'gray', None]

            for i, img in enumerate(images_to_plot):
                plt.subplot(1, len(images_to_plot), i + 1)
                if cmaps[i]: plt.imshow(img, cmap=cmaps[i])
                else: plt.imshow(img)
                plt.title(titles[i]); plt.axis('off')
                # Add colorbars to heatmaps
                if i in [3, 4, 5]: plt.colorbar(fraction=0.046, pad=0.04)

            plt.tight_layout(); plt.show()

        # --- Return Results ---
        # Return original resized PIL and outlined PIL (or original if no contours drawn)
        outlined_pil_image = Image.fromarray(cv2.cvtColor(outlined_image_np, cv2.COLOR_BGR2RGB)) # Convert back to RGB PIL
        return original_img_resized_pil, outlined_pil_image

    except Exception as e:
        print(f"Error during enhanced tumor outlining: {e}")
        print(traceback.format_exc())
        # Return None or original image on error
        try:
             return original_img_resized_pil, original_img_resized_pil
        except NameError: # If resizing failed early
             return None, None


# --- 7. Prediction and Outlining (Modified to use new outline function) ---

def predict_and_outline_simple(input_image_path,
                            classification_model_path='tumor_classification_densenet121.pth',
                            autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                            image_size=(256, 256),
                            error_threshold=None,
                            min_solidity=0.6, # Pass solidity threshold
                            normalize_transform=None, # For classifier
                            device_classifier=get_device(),
                            device_outlining=None, # Can be different (e.g., CPU)
                            show_intermediate_plots=False, # Control AE plots visibility
                            show_final_plot=True): # Control final comparison plot
    """Predicts tumor type and outlines using respective models."""
    print(f"\n--- Starting Prediction & Outlining for: {os.path.basename(input_image_path)} ---")
    if device_outlining is None: device_outlining = torch.device("cpu") # Default outlining to CPU
    if normalize_transform is None:
         print("Warning: Normalization transform not provided for classifier. Using default [0.5, 0.5].")
         # Make sure it's a transform object
         normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    original_resized_pil, outlined_final_image_pil = None, None # Initialize

    try:
        # --- Load Original Image ---
        original_img_pil = Image.open(input_image_path).convert('L')

        # --- Prepare Image for Classifier (includes normalization) ---
        transform_pipeline_classify = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
            normalize_transform # Apply classifier normalization
        ])
        img_tensor_classify = transform_pipeline_classify(original_img_pil).unsqueeze(0).to(device_classifier)

        # --- Classification ---
        print("Loading and running classifier...")
        loaded_classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False) # Instantiate correct class
        if not os.path.exists(classification_model_path):
            print(f"Error: Classifier model not found: {classification_model_path}"); return None, None, None
        loaded_classifier.load_state_dict(torch.load(classification_model_path, map_location=device_classifier))
        loaded_classifier.to(device_classifier).eval()
        class_map = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

        with torch.no_grad():
            prediction = loaded_classifier(img_tensor_classify)
            probabilities = torch.softmax(prediction, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)
            predicted_class_index = predicted_class_index.item()
            confidence = confidence.item()
        predicted_class_name = class_map.get(predicted_class_index, "Unknown")
        print(f"Predicted Tumor Type: {predicted_class_name} (Confidence: {confidence:.4f})")

        # --- Outlining ---
        if predicted_class_name != 'notumor':
            print("Tumor class predicted, preparing for outlining...")
            # --- Prepare Image for Autoencoder (NO normalization) ---
            transform_pipeline_ae = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Lambda(advanced_preprocess),
                transforms.ToTensor(), # Scales to [0, 1]
                transforms.Lambda(to_float_tensor),
            ])
            img_tensor_ae_batch = transform_pipeline_ae(original_img_pil).unsqueeze(0).to(device_outlining)

            # --- Load and Run Autoencoder ---
            print("Loading and running autoencoder...")
            loaded_autoencoder = EnhancedAutoencoder() # Instantiate correct class
            if not os.path.exists(autoencoder_model_path):
                 print(f"Error: Autoencoder model not found at {autoencoder_model_path}. Cannot outline.")
                 # Fallback: show original image as result
                 original_resized_pil = original_img_pil.resize(image_size)
                 outlined_final_image_pil = original_resized_pil
            else:
                loaded_autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device_outlining))
                loaded_autoencoder.to(device_outlining).eval()

                with torch.no_grad():
                    reconstructed_tensor_batch, attention_maps_tensor_batch = loaded_autoencoder(img_tensor_ae_batch)

                # --- Call Enhanced Outlining Function ---
                # Pass tensors without batch dim, on CPU for numpy processing
                original_resized_pil, outlined_final_image_pil = outline_tumor_enhanced(
                    original_image_pil=original_img_pil, # Original PIL for reference
                    input_tensor_ae=img_tensor_ae_batch.squeeze(0).cpu(),
                    reconstructed_tensor_ae=reconstructed_tensor_batch.squeeze(0).cpu(),
                    attention_map_tensor_ae=attention_maps_tensor_batch.squeeze(0).cpu(),
                    image_size=image_size,
                    error_threshold=error_threshold,
                    min_solidity=min_solidity,
                    show_plots=show_intermediate_plots # Control AE plots
                )
                if outlined_final_image_pil is None: # Handle outlining error
                    print("Outlining function returned an error.")
                    original_resized_pil = original_img_pil.resize(image_size)
                    outlined_final_image_pil = original_resized_pil # Show original on error

        else:
            print("No tumor class predicted. Skipping outlining.")
            original_resized_pil = original_img_pil.resize(image_size)
            outlined_final_image_pil = original_resized_pil # Show original if no tumor

        # --- Display Final Result ---
        if show_final_plot and original_resized_pil and outlined_final_image_pil:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(original_resized_pil, cmap='gray')
            plt.title('Original MRI (Resized)'); plt.axis('off')

            plt.subplot(1, 2, 2);
            # Check if outlining actually happened by comparing PIL images
            is_outlined = False
            if predicted_class_name != 'notumor' and outlined_final_image_pil is not None:
                 # Compare numpy arrays for content equality
                 if not np.array_equal(np.array(original_resized_pil), np.array(outlined_final_image_pil)):
                     is_outlined = True

            # Display the result (which might be the outlined image or the original)
            plt.imshow(np.array(outlined_final_image_pil)) # Show whatever the result is

            # Set title based on prediction and outlining success
            if predicted_class_name != 'notumor':
                 status = "(Outlined)" if is_outlined else "(Outline Failed/Not Applied)"
                 title = f'{predicted_class_name.capitalize()} Detected {status}\nConfidence: {confidence:.2%}'
            else:
                 title = f'No Tumor Detected\nConfidence: {confidence:.2%}'
            plt.title(title); plt.axis('off')

            plt.tight_layout(); plt.show()

        # Return prediction, confidence, and the final PIL image (outlined or original)
        return predicted_class_name, confidence, outlined_final_image_pil

    except FileNotFoundError:
        print(f"Error: Input image not found at {input_image_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during prediction and outlining: {e}")
        print(traceback.format_exc())
        return None, None, None


# --- 8. Main Execution (Adjusted parameters, paths, function calls) ---

def main():
    # freeze_support() # Uncomment if using multiprocessing on Windows/macOS

    # --- Configuration ---
    device = get_device()
    data_dir = 'E:\\project\\dataset1\\training1' # <<< --- CHECK/CHANGE THIS PATH
    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}"); return

    image_size = (256, 256)
    batch_size = 16 # Smaller batch size might be needed with CBAM/SSIM
    num_epochs_classifier = 25 # Might need more epochs
    num_epochs_autoencoder = 25 # AE might need more epochs with SSIM/Att loss
    num_workers = 2 # Adjust based on system (start lower)
    classifier_model_path = 'tumor_classification_densenet121_cbam_best.pth' # New name
    autoencoder_model_path = 'enhanced_autoencoder_notumor_cbam_ssim_v2_best.pth' # New name

    # AE Training Params
    ae_attention_weight = 0.03 # Tunable weight for attention sparsity loss
    ae_ssim_weight = 0.20 # Tunable weight for SSIM vs MSE loss (0.8 MSE + 0.2 SSIM)

    # Outlining Params
    outline_min_solidity = 0.65 # Minimum solidity for contour filtering (Tunable)

    # --- Data Loading & Preparation ---
    print("\n--- Loading Datasets ---")
    # Load full dataset *without* normalization first for mean/std calc
    full_dataset_for_stats = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=False)
    if len(full_dataset_for_stats) == 0: print("Error: No images found. Exiting."); return

    try:
        labels_list = full_dataset_for_stats.labels
        # Ensure dataset is large enough for stratified split
        min_class_count = min(Counter(labels_list).values()) if labels_list else 0
        test_val_size = 0.3 # 30% for validation + test
        test_size_from_temp = 0.5 # 50% of the 30% -> 15% test
        if min_class_count < 2: # Need at least 2 samples per class for stratification usually
             print(f"Warning: Smallest class has {min_class_count} samples. Using non-stratified split.")
             train_idx, temp_idx = train_test_split(list(range(len(full_dataset_for_stats))), test_size=test_val_size, random_state=42)
             val_idx, test_idx = train_test_split(temp_idx, test_size=test_size_from_temp, random_state=42)
        else:
            train_idx, temp_idx = train_test_split(list(range(len(full_dataset_for_stats))), test_size=test_val_size, random_state=42, stratify=labels_list)
            # Stratify validation/test split based on the temporary set's labels
            temp_labels = [labels_list[i] for i in temp_idx]
            min_temp_class_count = min(Counter(temp_labels).values()) if temp_labels else 0
            if min_temp_class_count < 2:
                 print(f"Warning: Smallest class in val/test split has {min_temp_class_count} samples. Using non-stratified val/test split.")
                 val_idx, test_idx = train_test_split(temp_idx, test_size=test_size_from_temp, random_state=42)
            else:
                val_idx, test_idx = train_test_split(temp_idx, test_size=test_size_from_temp, random_state=42, stratify=temp_labels)

    except ValueError as e:
        print(f"Error during data splitting: {e}. Check class distribution and dataset size."); return

    # Calculate Mean/Std on Training Data Subset (using the *non-normalized* dataset)
    print(f"Using {len(train_idx)} samples for mean/std calculation.")
    temp_train_dataset = torch.utils.data.Subset(full_dataset_for_stats, train_idx)
    # Use a temporary loader, ensure shuffle=False for calc consistency if needed
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_mean, train_std = calculate_mean_std(temp_train_loader)
    print(f"Calculated Mean: {train_mean.item():.4f}, Std: {train_std.item():.4f}")
    global normalize_transform # Make accessible globally
    normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)

    # Create Final Datasets (Applying normalization transform where needed)
    # Classifier dataset: uses normalization
    full_classification_dataset = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=True)
    full_classification_dataset.set_normalization(normalize_transform) # Set the calculated norm
    train_classification_dataset = torch.utils.data.Subset(full_classification_dataset, train_idx)
    val_classification_dataset = torch.utils.data.Subset(full_classification_dataset, val_idx)
    test_classification_dataset = torch.utils.data.Subset(full_classification_dataset, test_idx) # Keep test set separate

    # DataLoaders for Classifier
    pin_memory = True if device.type != 'cpu' else False
    # persistent_workers can cause issues sometimes, disable if dataloading hangs
    persistent_workers = (num_workers > 0)
    train_classification_loader = DataLoader(train_classification_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=True) # Drop last might help with batchnorm stability
    val_classification_loader = DataLoader(val_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_classification_loader = DataLoader(test_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    # DataLoader for Autoencoder (Uses *non-normalized* NotumorMRIDataset)
    notumor_autoencoder_dataset = NotumorMRIDataset(data_dir, image_size=image_size, split='train')
    train_autoencoder_loader = None
    if len(notumor_autoencoder_dataset) > 0:
        train_autoencoder_loader = DataLoader(
            notumor_autoencoder_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers, drop_last=True # Drop last might help AE stability
            )
    else:
        print("Warning: No 'notumor' images found for AE training.")

    # --- Autoencoder Training ---
    if train_autoencoder_loader:
        print("\n--- Training Enhanced Autoencoder with CBAM & SSIM Loss ---")
        autoencoder = EnhancedAutoencoder(cbam_reduction_ratio=8, cbam_kernel_size=5).to(device)
        trained_autoencoder = train_enhanced_autoencoder(
            autoencoder, train_autoencoder_loader, num_epochs=num_epochs_autoencoder,
            device=device, model_path=autoencoder_model_path,
            attention_weight=ae_attention_weight, # Pass tunable weight
            ssim_weight=ae_ssim_weight       # Pass tunable weight
        )
    else:
        print("\n--- Skipping Autoencoder Training (No Data/Loader) ---")
        if not os.path.exists(autoencoder_model_path):
            print(f"Warning: AE model not found at {autoencoder_model_path} and cannot be trained. Outlining might fail.")

    # --- Classifier Training ---
    print("\n--- Training Classifier with CBAM ---")
    classifier = DenseNet121TumorClassificationModel(
        num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7
    ).to(device)

    # Loss, Optimizer, Scheduler for Classifier
    # Recalculate weights based on the actual training indices used
    train_labels = [full_classification_dataset.labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    weights = torch.ones(4, dtype=torch.float) # Default weights
    if len(class_counts) == 4 and all(c > 0 for c in class_counts.values()):
        # Calculate weights based on inverse frequency, normalized
        total_samples = len(train_labels)
        weights_arr = np.array([total_samples / class_counts[i] for i in range(4)])
        # Normalize weights
        weights_arr /= np.sum(weights_arr)
        # Scale (optional, often helps balance with typical loss magnitudes)
        weights = torch.tensor(weights_arr * 4, dtype=torch.float).to(device)
        print(f"Using Calculated Class Weights: {weights.cpu().numpy()}")
    else:
        print(f"Warning: Could not calculate accurate weights (Counts: {class_counts}). Using default weights [1,1,1,1].")
        weights = weights.to(device)

    criterion_classifier = nn.CrossEntropyLoss(weight=weights)
    # Consider different LR for backbone and head if fine-tuning
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=5e-5, weight_decay=1e-4) # Lower LR for fine-tuning
    total_steps = num_epochs_classifier * len(train_classification_loader)
    scheduler_classifier = OneCycleLR(optimizer_classifier, max_lr=5e-4, total_steps=total_steps, pct_start=0.25, anneal_strategy='cos', div_factor=10, final_div_factor=1e4)
    # Alternative scheduler:
    # scheduler_classifier = ReduceLROnPlateau(optimizer_classifier, mode='max', factor=0.2, patience=4, verbose=True) # Step based on F1

    trained_classifier = train_classifier(
        classifier, train_classification_loader, val_classification_loader,
        criterion_classifier, optimizer_classifier, scheduler=scheduler_classifier,
        num_epochs=num_epochs_classifier, device=device, model_path=classifier_model_path
    )

    # --- Prediction and Outlining on a Sample Image ---
    print("\n--- Performing Prediction and Outlining on a Sample Image ---")
    # <<< --- CHECK/CHANGE THIS PATH to a valid image --- >>>
    sample_image_path ="C:\\Users\\Harish\\Downloads\\PituitaryAdenoma.jpeg" # Example

    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found: {sample_image_path}")
        # Try to get one from the test set
        test_image_path = None
        if test_idx and hasattr(full_classification_dataset, 'image_paths') and len(test_idx) > 0:
             try: test_image_path = full_classification_dataset.image_paths[test_idx[0]]
             except IndexError: test_image_path = None
             if test_image_path and os.path.exists(test_image_path):
                  sample_image_path = test_image_path
                  print(f"Using sample from test set: {sample_image_path}")
             else: sample_image_path = None # Reset if path from test set is invalid
        else: sample_image_path = None

        if not sample_image_path:
             print("Cannot find a suitable sample image for prediction demo.")

    if sample_image_path:
        pred_class, confidence, final_image = predict_and_outline_simple(
            input_image_path=sample_image_path,
            classification_model_path=classifier_model_path,
            autoencoder_model_path=autoencoder_model_path,
            image_size=image_size,
            error_threshold=None, # Use Otsu by default
            min_solidity=outline_min_solidity, # Pass solidity threshold
            normalize_transform=normalize_transform, # Crucial for classifier
            device_classifier=device,
            device_outlining=torch.device("cpu"), # Outline on CPU is usually fine
            show_intermediate_plots=True, # Show the detailed AE/outlining plots
            show_final_plot=True         # Show the final comparison plot
        )
        if pred_class is not None:
            print(f"\nFinal Prediction for Sample: {pred_class} (Confidence: {confidence*100:.2f}%)")
            # Optionally save the final image
            # if final_image:
            #     save_path = f"./{os.path.basename(sample_image_path).split('.')[0]}_outlined.png"
            #     final_image.save(save_path)
            #     print(f"Saved outlined image to {save_path}")
    else:
        print("Skipping final prediction demo as no sample image was found.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    # Call freeze_support() at the very beginning if using multiprocessing
    freeze_support()
    main()