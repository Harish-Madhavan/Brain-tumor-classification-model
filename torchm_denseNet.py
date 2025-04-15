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

# --- CBAM Components ---

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            # Use Conv2d with kernel_size 1 as equivalent to Linear layers applied spatially
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att # Apply attention: element-wise multiplication

class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # Convolution layer to process pooled features
        self.conv = nn.Conv2d(
            in_channels=2, # Takes concatenated avg-pooled and max-pooled features
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool features along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True) # Average across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Max across channels
        # Concatenate pooled features
        pooled_features = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and activation to get spatial attention map
        spatial_att = self.sigmoid(self.conv(pooled_features))
        return x * spatial_att # Apply attention: element-wise multiplication

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x

# --- End CBAM Components ---


# --- Helper Functions (including preprocessing, defined globally) ---

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
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return Image.fromarray(img_blur)

def to_float_tensor(img):
    """Convert a tensor to float."""
    return img.float()

# --- Data Loading and Preprocessing ---

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
        self.base_transform_list = [
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ]
        self.normalize_transform = None
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
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
            current_transforms = list(self.base_transform_list)
            base_pipeline = transforms.Compose(current_transforms)
            image = base_pipeline(image_pil)
            if self.split == 'train':
                image = self.augment_transforms(image)
            if self.apply_normalize and self.normalize_transform:
                 image = self.normalize_transform(image)
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading or processing image: {image_path}, error: {e}")
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.tensor(-1, dtype=torch.long)

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
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ])
        print(f"{split} 'notumor' data loading complete. Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = self.transforms(image_pil)
            return image, image # Target is the image itself
        except Exception as e:
            print(f"Error loading or processing 'notumor' image: {image_path}, error: {e}")
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))

# --- 2. Autoencoder Model with CBAM ---

class EnhancedAutoencoder(nn.Module):
    def __init__(self, cbam_reduction_ratio=8, cbam_kernel_size=5): # CBAM params for AE
        super(EnhancedAutoencoder, self).__init__()
        print(f"Initializing Enhanced Autoencoder with CBAM (reduction={cbam_reduction_ratio}, kernel={cbam_kernel_size})")
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

        # Original attention branch (operates on CBAM-refined bottleneck features)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_dec3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_dec2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_dec1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1) # Final output from refined dec1 features

    def forward(self, x):
        # --- Encoder (Save raw outputs for skip connections) ---
        e1_raw = self.enc1(x)
        e1 = self.cbam_enc1(e1_raw) # Apply CBAM
        p1 = self.pool1(e1)

        e2_raw = self.enc2(p1)
        e2 = self.cbam_enc2(e2_raw) # Apply CBAM
        p2 = self.pool2(e2)

        e3_raw = self.enc3(p2)
        e3 = self.cbam_enc3(e3_raw) # Apply CBAM
        p3 = self.pool3(e3)

        # --- Bottleneck ---
        b_raw = self.bottleneck(p3)
        b = self.cbam_bottleneck(b_raw) # Apply CBAM

        # Calculate attention map from CBAM-refined bottleneck features
        att_map = self.attention(b)

        # --- Decoder ---
        d3_up = self.up3(b)
        # Concatenate with *original* encoder feature (e3_raw)
        d3_cat = torch.cat([d3_up, e3_raw], dim=1)
        d3_processed = self.dec3(d3_cat)
        d3 = self.cbam_dec3(d3_processed) # Apply CBAM after processing

        d2_up = self.up2(d3)
        d2_cat = torch.cat([d2_up, e2_raw], dim=1)
        d2_processed = self.dec2(d2_cat)
        d2 = self.cbam_dec2(d2_processed) # Apply CBAM

        d1_up = self.up1(d2)
        d1_cat = torch.cat([d1_up, e1_raw], dim=1)
        d1_processed = self.dec1(d1_cat)
        d1 = self.cbam_dec1(d1_processed) # Apply CBAM

        # --- Output ---
        reconstruction = torch.sigmoid(self.out_conv(d1))

        return reconstruction, att_map

# --- 3. DenseNet121 Model with CBAM for Classification ---

class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7):
        super(DenseNet121TumorClassificationModel, self).__init__()
        print(f"Initializing DenseNet121 Classifier with CBAM (reduction={cbam_reduction_ratio}, kernel={cbam_kernel_size})")

        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)

        # Modify first conv layer for grayscale
        original_conv0 = self.densenet121.features.conv0
        self.densenet121.features.conv0 = nn.Conv2d(1, original_conv0.out_channels,
                                                   kernel_size=original_conv0.kernel_size, stride=original_conv0.stride,
                                                   padding=original_conv0.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                self.densenet121.features.conv0.weight.copy_(original_conv0.weight.mean(dim=1, keepdim=True))

        # Get number of features from the backbone
        num_ftrs = self.densenet121.classifier.in_features

        # *** Add CBAM layer after the feature extractor ***
        self.cbam = CBAM(num_ftrs, reduction_ratio=cbam_reduction_ratio, kernel_size=cbam_kernel_size)
        print(f"Added CBAM block after DenseNet features with {num_ftrs} channels.")

        # Replace the original classifier with a custom head
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        # Remove the original DenseNet classifier part
        self.densenet121.classifier = nn.Identity()

        self._initialize_weights(self.classifier_head)

    def _initialize_weights(self, classifier_module):
        for m in classifier_module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Extract features
        features = self.densenet121.features(x)
        # 2. Apply CBAM
        features = self.cbam(features)
        # 3. Global Pooling and Classification Head
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier_head(out)
        return out

# --- 4. Utility Functions ---
def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_samples = 0
    num_batches = 0
    print("Calculating mean and std...")
    for images, _ in tqdm(loader, desc="Mean/Std Calc"):
        if not isinstance(images, torch.Tensor) or images.nelement() == 0 or images.shape[1] != 1:
             continue
        batch_samples = images.size(0)
        images_flat = images.view(batch_samples, images.size(1), -1)
        mean += images_flat.mean(2).sum(0)
        std += images_flat.std(2).sum(0)
        total_samples += batch_samples
        num_batches += 1
    if total_samples == 0:
        print("Warning: No valid samples found for mean/std calculation. Using defaults [0.5, 0.5].")
        return torch.tensor([0.5]), torch.tensor([0.5])
    mean /= total_samples
    std /= total_samples
    std = torch.max(std, torch.tensor(1e-6)) # Prevent division by zero
    return mean, std

# --- 5. Training Loops ---

def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path='enhanced_autoencoder_notumor.pth', attention_weight=0.1):
    if os.path.exists(model_path):
        print(f"Loading existing autoencoder from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Autoencoder loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading autoencoder model: {e}. Training from scratch.")

    model = model.to(device)
    criterion_recon = nn.MSELoss() # Reconstruction loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    print(f"--- Starting Enhanced Autoencoder Training ({num_epochs} epochs) ---")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss, recon_loss_epoch, att_loss_epoch = 0.0, 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [AE Train]")

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            reconstructed, attention_maps = model(images)

            loss_recon = criterion_recon(reconstructed, targets)
            # Attention loss (L1 penalty on attention map) - encourage sparsity/focus
            loss_att = torch.mean(torch.abs(attention_maps))
            loss = loss_recon + attention_weight * loss_att # Combined loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            recon_loss_epoch += loss_recon.item()
            att_loss_epoch += loss_att.item()
            progress_bar.set_postfix(Loss=f"{loss.item():.4f}", ReconL=f"{loss_recon.item():.4f}", AttL=f"{loss_att.item():.4f}")

        avg_total_loss = total_loss / len(train_loader)
        avg_recon_loss = recon_loss_epoch / len(train_loader)
        avg_att_loss = att_loss_epoch / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Completed: Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Atten Loss: {avg_att_loss:.4f}")
        scheduler.step(avg_total_loss) # Step LR scheduler

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best AE model saved to {model_path} (Loss: {best_loss:.4f})")

    print("--- Enhanced Autoencoder training complete. ---")
    try: # Reload best model
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best autoencoder model from {model_path}")
    except Exception as e:
         print(f"Warning: Could not reload best AE model after training: {e}")
    return model


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
            if torch.any(labels == -1): continue # Skip error batches
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
            progress_bar_train.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_train_loss = train_loss / len(train_loader)
        # Calculate training metrics (use try-except for safety if lists are empty)
        try:
            train_accuracy = 100 * accuracy_score(all_labels_train, all_preds_train)
            train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        except ValueError:
            train_accuracy, train_f1 = 0.0, 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_labels_val, all_preds_val = [], []
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in progress_bar_val:
                if torch.any(labels == -1): continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(predicted.cpu().numpy())
                progress_bar_val.set_postfix(Loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        try:
            val_accuracy = 100 * accuracy_score(all_labels_val, all_preds_val)
            val_precision = precision_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
            val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        except ValueError:
             val_accuracy, val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0, 0.0


        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")
        print(f"  Val P/R: {val_precision:.4f}/{val_recall:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Step ReduceLROnPlateau scheduler based on validation F1
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_f1) # Step based on F1 score

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  New best classifier model saved to {model_path} (Val F1: {best_val_f1:.4f})")

    print("--- Classifier training complete. ---")
    try: # Reload best model
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best classifier model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not reload best classifier model after training: {e}")
    return model

# --- 6. Enhanced Tumor Outlining Function ---

def enhance_outline_detection(reconstruction_error, attention_map, threshold=None):
    """Combine maps, filter, threshold, and refine using morphology."""
    # Normalize maps robustly
    rec_min, rec_max = reconstruction_error.min(), reconstruction_error.max()
    att_min, att_max = attention_map.min(), attention_map.max()
    rec_norm = (reconstruction_error - rec_min) / (rec_max - rec_min + 1e-8)
    att_norm = (attention_map - att_min) / (att_max - att_min + 1e-8)

    # Combine (Weight attention slightly higher?)
    combined_map = np.clip((0.4 * rec_norm + 0.6 * att_norm), 0, 1)
    combined_uint8 = (combined_map * 255).astype(np.uint8)

    # Filtering (Median is often good for salt-and-pepper noise)
    combined_gauss = cv2.GaussianBlur(combined_uint8, (5, 5), 0)
    combined_median = cv2.medianBlur(combined_gauss, 5)
    combined_filtered_uint8 = combined_median
    combined_filtered = combined_filtered_uint8.astype(np.float32) / 255.0

    # Thresholding
    if threshold is None:
        if combined_filtered_uint8.max() > 0:
            otsu_thresh_val = threshold_otsu(combined_filtered_uint8)
            threshold = otsu_thresh_val / 255.0
            print(f"  Otsu threshold determined: {threshold:.4f} (Value: {otsu_thresh_val})")
        else:
            threshold = 0.99 # High threshold if map is black
            print("  Warning: Filtered map is all black, using default high threshold.")
    else:
        print(f"  Using provided threshold: {threshold:.4f}")
    tumor_mask = (combined_filtered > threshold).astype(np.uint8)

    # Morphological Refinement
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel_ellipse_5, iterations=2)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_ellipse_3, iterations=2)
    tumor_mask_refined = mask_opened

    return tumor_mask_refined, combined_filtered


def outline_tumor_enhanced(input_image_path, autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                           image_size=(224, 224), error_threshold=None, device=torch.device("cpu"),
                           show_plots=True):
    """Outlines tumor using Enhanced AE (potentially with CBAM)"""
    print(f"\n--- Starting Tumor Outlining for: {os.path.basename(input_image_path)} ---")
    print(f"Using AE model: {autoencoder_model_path} on device: {device}")
    try:
        # --- Load Model ---
        # Instantiate the *correct* AE class (assuming it might have CBAM args now)
        loaded_autoencoder = EnhancedAutoencoder() # Add CBAM args if needed for loading state_dict
        if not os.path.exists(autoencoder_model_path):
             print(f"Error: Autoencoder model not found at {autoencoder_model_path}")
             return None, None
        loaded_autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
        loaded_autoencoder.to(device).eval()
        print("Autoencoder model loaded.")

        # --- Load & Prepare Image ---
        original_img_pil = Image.open(input_image_path).convert('L')
        original_img_resized_pil = original_img_pil.resize(image_size)
        original_img_resized_np = np.array(original_img_resized_pil)

        # AE Preprocessing (NO normalization)
        preprocessed_pil = advanced_preprocess(original_img_pil) # Preprocess before resize
        transform_pipeline_ae = transforms.Compose([
            transforms.Resize(image_size), # Resize after advanced_preprocess
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ])
        img_tensor_ae = transform_pipeline_ae(preprocessed_pil).unsqueeze(0).to(device)

        # --- AE Inference ---
        with torch.no_grad():
            reconstructed_img_tensor, attention_maps_tensor = loaded_autoencoder(img_tensor_ae)

        # --- Process Outputs ---
        reconstructed_img_np = reconstructed_img_tensor.cpu().squeeze().numpy()
        attention_map_np = attention_maps_tensor.cpu().squeeze().numpy() # Size depends on AE arch [H/8, W/8] likely
        input_img_ae_np = img_tensor_ae.cpu().squeeze().numpy() # AE input [0,1]

        # --- Calculate Error & Upsample Attention ---
        reconstruction_error = np.abs(input_img_ae_np - reconstructed_img_np)
        attention_map_resized = cv2.resize(attention_map_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

        # --- Generate Mask ---
        print("Running enhanced outline detection...")
        tumor_mask, combined_map_filtered = enhance_outline_detection(
            reconstruction_error, attention_map_resized, threshold=error_threshold)

        # --- Find & Filter Contours ---
        tumor_mask_uint8 = tumor_mask.astype(np.uint8)
        contours, _ = cv2.findContours(tumor_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Number of contours found: {len(contours)}")

        # --- Create Outlined Image ---
        outlined_image_np = cv2.cvtColor(original_img_resized_np, cv2.COLOR_GRAY2BGR)
        overlay = outlined_image_np.copy()
        contours_drawn = 0
        if contours:
            min_area, max_area = 50, (image_size[0] * image_size[1] * 0.6)
            valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
            print(f"Valid contours after area filtering: {len(valid_contours)}")
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3] # Top 3

            for contour in valid_contours:
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), cv2.FILLED) # Green fill on overlay
                cv2.drawContours(outlined_image_np, [contour], -1, (255, 0, 0), 2) # Red outline on image
                contours_drawn += 1

            alpha = 0.3 # Blend overlay
            cv2.addWeighted(overlay, alpha, outlined_image_np, 1 - alpha, 0, outlined_image_np)
            print(f"Drawn {contours_drawn} contours.")

        # --- Plotting ---
        if show_plots:
            plt.figure(figsize=(18, 6))
            titles = ['Original (Resized)', 'Recon Error', 'Attention Map (Resized)', 'Final Mask', 'Outlined Result']
            images = [original_img_resized_np, reconstruction_error, attention_map_resized, tumor_mask_uint8, cv2.cvtColor(outlined_image_np, cv2.COLOR_BGR2RGB)]
            cmaps = ['gray', 'hot', 'viridis', 'gray', None]
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                if cmaps[i]: plt.imshow(images[i], cmap=cmaps[i])
                else: plt.imshow(images[i])
                plt.title(titles[i]); plt.axis('off')
                if i in [1, 2]: plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout(); plt.show()

        # --- Return Results ---
        # Return original resized PIL and outlined PIL (or original if no contours drawn)
        outlined_pil_image = Image.fromarray(outlined_image_np) if contours_drawn > 0 else original_img_resized_pil
        return original_img_resized_pil, outlined_pil_image

    except Exception as e:
        print(f"Error during enhanced tumor outlining: {e}")
        print(traceback.format_exc())
        return None, None

# --- 7. Prediction and Outlining ---

def predict_and_outline_simple(input_image_path,
                            classification_model_path='tumor_classification_densenet121.pth',
                            autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                            image_size=(256, 256),
                            error_threshold=None,
                            normalize_transform=None,
                            device_classifier=get_device(),
                            device_outlining=None,
                            show_plots=True):
    """Predicts tumor type and outlines using respective models (potentially CBAM versions)."""
    print(f"\n--- Starting Prediction & Outlining for: {os.path.basename(input_image_path)} ---")
    if device_outlining is None: device_outlining = torch.device("cpu")
    if normalize_transform is None:
         print("Warning: Normalization transform not provided. Using default [0.5, 0.5].")
         normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    try:
        # --- Load Classifier ---
        # Instantiate the *correct* classifier class (assuming it might have CBAM args)
        loaded_classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False) # Add CBAM args if used
        if not os.path.exists(classification_model_path):
            print(f"Error: Classifier model not found: {classification_model_path}"); return None, None
        loaded_classifier.load_state_dict(torch.load(classification_model_path, map_location=device_classifier))
        loaded_classifier.to(device_classifier).eval()
        class_map = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
        print("Classifier model loaded.")

        # --- Prepare Image for Classifier ---
        original_img_pil = Image.open(input_image_path).convert('L')
        transform_pipeline_classify = transforms.Compose([ # Classifier pipeline (includes normalization)
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
            normalize_transform # Apply classifier normalization
        ])
        img_tensor_classify = transform_pipeline_classify(original_img_pil).unsqueeze(0).to(device_classifier)

        # --- Classification ---
        with torch.no_grad():
            prediction = loaded_classifier(img_tensor_classify)
            probabilities = torch.softmax(prediction, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)
            predicted_class_index = predicted_class_index.item()
            confidence = confidence.item()
        predicted_class_name = class_map.get(predicted_class_index, "Unknown")
        print(f"Predicted Tumor Type: {predicted_class_name} (Confidence: {confidence:.4f})")

        # --- Outlining ---
        original_resized_pil, outlined_final_image_pil = None, None
        if predicted_class_name != 'notumor':
            print("Tumor class predicted, attempting outlining...")
            original_resized_pil, outlined_final_image_pil = outline_tumor_enhanced(
                input_image_path=input_image_path, autoencoder_model_path=autoencoder_model_path,
                image_size=image_size, error_threshold=error_threshold,
                device=device_outlining, show_plots=False # Control AE plot from here
            )
            if outlined_final_image_pil is None: # Handle outlining error
                 print("Outlining function returned an error.")
                 original_resized_pil = original_img_pil.resize(image_size)
                 outlined_final_image_pil = original_resized_pil # Show original on error
        else:
            print("No tumor class predicted.")
            original_resized_pil = original_img_pil.resize(image_size)
            outlined_final_image_pil = original_resized_pil # Show original if no tumor

        # --- Display Final Result ---
        if show_plots and original_resized_pil and outlined_final_image_pil:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(original_resized_pil, cmap='gray')
            plt.title('Original MRI (Resized)'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(np.array(outlined_final_image_pil))
            title = 'Result'
            if predicted_class_name != 'notumor':
                 is_outlined = original_resized_pil is not outlined_final_image_pil
                 status = "(Outlined)" if is_outlined else "(Outline Failed)"
                 title = f'{predicted_class_name.capitalize()} Detected {status}\nConfidence: {confidence:.2%}'
            else: title = f'No Tumor Detected\nConfidence: {confidence:.2%}'
            plt.title(title); plt.axis('off')
            plt.tight_layout(); plt.show()

        return predicted_class_name, confidence

    except Exception as e:
        print(f"An error occurred during prediction and outlining: {e}")
        print(traceback.format_exc())
        return None, None

# --- 8. Main Execution ---

def main():
    # freeze_support() # Uncomment if using multiprocessing on Windows/macOS

    # --- Configuration ---
    device = get_device()
    data_dir = 'E:\\project\\dataset1\\training1' # <<< --- CHECK/CHANGE THIS PATH
    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}"); return

    image_size = (256, 256) # Resize for training and inference
    batch_size = 32 # Reduce if OOM with CBAM
    num_epochs_classifier = 25
    num_epochs_autoencoder = 20 # May need more epochs with CBAM
    num_workers = 4 # Adjust based on system
    classifier_model_path = 'tumor_classification_densenet121_cbam_best.pth' # New name
    autoencoder_model_path = 'enhanced_autoencoder_notumor_cbam_best.pth' # New name

    # --- Data Loading & Preparation ---
    print("\n--- Loading Datasets ---")
    full_dataset_for_stats = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=False)
    if len(full_dataset_for_stats) == 0: print("Error: No images found. Exiting."); return

    try: # Data Splitting
        labels_list = full_dataset_for_stats.labels
        train_idx, temp_idx = train_test_split(list(range(len(full_dataset_for_stats))), test_size=0.3, random_state=42, stratify=labels_list)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[labels_list[i] for i in temp_idx])
    except ValueError as e:
        print(f"Error during split (maybe small classes?): {e}"); return

    # Calculate Mean/Std on Training Data (No Norm applied yet)
    temp_train_dataset = torch.utils.data.Subset(full_dataset_for_stats, train_idx)
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_mean, train_std = calculate_mean_std(temp_train_loader)
    print(f"Calculated Mean: {train_mean.item():.4f}, Std: {train_std.item():.4f}")
    global normalize_transform
    normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)

    # Create Final Datasets (Now with normalization for classifier)
    full_classification_dataset = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=True)
    full_classification_dataset.set_normalization(normalize_transform)
    train_classification_dataset = torch.utils.data.Subset(full_classification_dataset, train_idx)
    val_classification_dataset = torch.utils.data.Subset(full_classification_dataset, val_idx)
    # test_classification_dataset = torch.utils.data.Subset(full_classification_dataset, test_idx) # Keep test set

    # DataLoaders for Classifier
    pin_memory = True if device.type != 'cpu' else False
    persistent_workers = (num_workers > 0)
    train_classification_loader = DataLoader(train_classification_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_classification_loader = DataLoader(val_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    # test_classification_loader = DataLoader(test_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    # DataLoader for Autoencoder (No Norm)
    notumor_autoencoder_dataset = NotumorMRIDataset(data_dir, image_size=image_size, split='train')
    train_autoencoder_loader = None
    if len(notumor_autoencoder_dataset) > 0:
        train_autoencoder_loader = DataLoader(notumor_autoencoder_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    else:
        print("Warning: No 'notumor' images found for AE training.")

    # --- Autoencoder Training ---
    if train_autoencoder_loader:
        print("\n--- Training Enhanced Autoencoder with CBAM ---")
        autoencoder = EnhancedAutoencoder(cbam_reduction_ratio=8, cbam_kernel_size=5).to(device) # Instantiate CBAM AE
        trained_autoencoder = train_enhanced_autoencoder(
            autoencoder, train_autoencoder_loader, num_epochs=num_epochs_autoencoder,
            device=device, model_path=autoencoder_model_path, attention_weight=0.05
        )
    else: # Try loading existing AE if training skipped
        print("\n--- Skipping Autoencoder Training (No Data/Loader) ---")
        if not os.path.exists(autoencoder_model_path):
            print(f"Warning: AE model not found at {autoencoder_model_path} and cannot be trained. Outlining might fail.")

    # --- Classifier Training ---
    print("\n--- Training Classifier with CBAM ---")
    classifier = DenseNet121TumorClassificationModel( # Instantiate CBAM Classifier
        num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7
    ).to(device)

    # Loss, Optimizer, Scheduler for Classifier
    class_counts = Counter(full_classification_dataset.labels)
    weights = torch.ones(4, dtype=torch.float)
    if len(class_counts) == 4:
        counts_arr = np.array([class_counts[i] for i in range(4)])
        weights_arr = 1.0 / counts_arr
        weights_arr = weights_arr / np.sum(weights_arr)
        weights = torch.tensor(weights_arr * 4, dtype=torch.float).to(device)
    else: print("Warning: Class imbalance weights might be inaccurate.")
    print(f"Using Class Weights: {weights.cpu().numpy()}")
    criterion_classifier = nn.CrossEntropyLoss(weight=weights)
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4) # Fine-tuning LR
    total_steps = num_epochs_classifier * len(train_classification_loader)
    scheduler_classifier = OneCycleLR(optimizer_classifier, max_lr=1e-3, total_steps=total_steps, pct_start=0.2, anneal_strategy='cos', div_factor=10, final_div_factor=1e4)
    # scheduler_classifier = ReduceLROnPlateau(optimizer_classifier, mode='max', factor=0.2, patience=5, verbose=True) # Alternative: step based on F1

    trained_classifier = train_classifier(
        classifier, train_classification_loader, val_classification_loader,
        criterion_classifier, optimizer_classifier, scheduler=scheduler_classifier,
        num_epochs=num_epochs_classifier, device=device, model_path=classifier_model_path
    )

    # --- Prediction and Outlining on a Sample Image ---
    print("\n--- Performing Prediction and Outlining on a Sample Image ---")
    # <<< --- CHECK/CHANGE THIS PATH to a valid image --- >>>
    sample_image_path = "E:\\project\\dataset1\\training1\\meningioma\\Te-me_0085.jpg"

    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found: {sample_image_path}")
        # Try one from test set?
        if test_idx and hasattr(full_classification_dataset, 'image_paths'):
             try: sample_image_path = full_classification_dataset.image_paths[test_idx[0]]
             except IndexError: sample_image_path = None
        else: sample_image_path = None
        if sample_image_path: print(f"Using sample from test set: {sample_image_path}")
        else: print("Cannot find a sample image for prediction.")

    if sample_image_path:
        predict_and_outline_simple(
            input_image_path=sample_image_path,
            classification_model_path=classifier_model_path, # Use CBAM model path
            autoencoder_model_path=autoencoder_model_path,   # Use CBAM model path
            image_size=image_size,
            error_threshold=None, # Use Otsu
            normalize_transform=normalize_transform, # Pass the calculated normalization
            device_classifier=device,
            device_outlining=torch.device("cpu"), # Outline on CPU
            show_plots=True
        )
    else:
        print("Skipping final prediction demo.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    freeze_support() # Call first if using multiprocessing
    main()