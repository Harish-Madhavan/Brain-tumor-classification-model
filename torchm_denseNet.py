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
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from skimage.filters import threshold_otsu
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau # Added ReduceLROnPlateau for clarity
from multiprocessing import freeze_support
import traceback # For detailed error printing

# --- Helper Functions (including preprocessing, defined globally) ---

def get_device():
    """Helper to get the best available device (Prioritizes XPU if available)"""
    if torch.xpu.is_available():
        print("Using XPU device.")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("XPU not available, using CUDA device.")
        return torch.device("cuda")
    else:
        print("XPU and CUDA not available, using CPU device.")
        return torch.device("cpu")

def equalize_histogram(img):
    """
    Apply histogram equalization to a grayscale PIL image.
    """
    return ImageOps.equalize(img)

def advanced_preprocess(img):
    """
    Apply advanced pre-processing by performing CLAHE (for adaptive histogram equalization)
    combined with a slight Gaussian blur for noise reduction on a grayscale PIL image.
    """
    # Convert PIL image to a NumPy array
    img_np = np.array(img)

    # Apply CLAHE for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)

    # Apply a Gaussian blur to reduce noise without losing too much detail
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Convert back to PIL image
    return Image.fromarray(img_blur)

def to_float_tensor(img):
    """Convert a tensor to float."""
    return img.float()

# --- Data Loading and Preprocessing ---

class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', apply_normalize=True): # Added apply_normalize flag
        self.image_paths = []
        self.labels = []
        class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_labels_map = {folder: i for i, folder in enumerate(class_folders)}
        self.class_counts = {folder: 0 for folder in class_folders}
        self.apply_normalize = apply_normalize # Store normalization preference

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

        # Base transforms (applied to all splits)
        self.base_transform_list = [
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ]
        # Defer adding normalization until it's calculated and set externally
        self.normalize_transform = None # Placeholder

        # --- Expanded Data Augmentation (only for training) ---
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=15
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1), # Note: Applied after ToTensor if used here
        ])

        print(f"{split} data loading complete. Found {len(self.image_paths)} images.")
        print(f"Class Distribution for {split} set: {self.class_counts}")

    def set_normalization(self, normalize_transform):
        """Sets the normalization transform after calculation."""
        self.normalize_transform = normalize_transform
        print("Normalization transform set for dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L')

            # Apply base transforms first
            current_transforms = list(self.base_transform_list) # Create a copy

            # Apply augmentations *before* ToTensor if they operate on PIL images
            if self.split == 'train':
                # Note: Most augmentations are better applied *after* ToTensor for consistency,
                # but PIL-based ones like flips/rotations can be here.
                # However, the current setup applies augmentations *after* ToTensor.
                pass # Augmentations are applied later in the current structure

            # Create the base transform pipeline
            base_pipeline = transforms.Compose(current_transforms)
            image = base_pipeline(image_pil) # Output is a Tensor [0, 1]

            # Apply augmentations (if train split) - operates on Tensor
            if self.split == 'train':
                image = self.augment_transforms(image)

            # Apply normalization if requested and set
            if self.apply_normalize and self.normalize_transform:
                 image = self.normalize_transform(image)

            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading or processing image: {image_path}, error: {e}")
            # Return a placeholder tensor
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.tensor(-1, dtype=torch.long) # Use -1 label for error


class NotumorMRIDataset(Dataset):
    """Dataset specifically for loading 'notumor' images for Autoencoder training."""
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
        self.split = split # Although primarily for 'train', keep for consistency
        # Define transforms *without* normalization, matching AE requirements
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
            # Autoencoder uses the image itself as the target
            return image, image
        except Exception as e:
            print(f"Error loading or processing 'notumor' image: {image_path}, error: {e}")
            # Return placeholder tensors
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))

# --- 2. Autoencoder Models ---

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2) # 256 -> 128
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2) # 128 -> 64

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # 64 -> 128
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2) # 128 -> 256

    def forward(self, x):
        # Encode
        x = F.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_pool2(x) # Bottleneck

        # Decode
        x = F.relu(self.dec_conv1(x))
        x = torch.sigmoid(self.dec_conv2(x)) # Output in [0, 1] range
        return x

class EnhancedAutoencoder(nn.Module):
    def __init__(self):
        super(EnhancedAutoencoder, self).__init__()
        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # H,W -> H/2,W/2

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # H/2,W/2 -> H/4,W/4

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # H/4,W/4 -> H/8,W/8

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Attention branch from bottleneck features
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1), # Produce a single channel map
            nn.Sigmoid() # Attention weights between 0 and 1
        )

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # H/8 -> H/4
        self.dec3 = nn.Sequential(
            # Concatenate skip connection from enc3 (128) + upsampled bottleneck (128) = 256
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # H/4 -> H/2
        self.dec2 = nn.Sequential(
            # Concatenate skip connection from enc2 (64) + upsampled dec3 (64) = 128
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # H/2 -> H
        self.dec1 = nn.Sequential(
            # Concatenate skip connection from enc1 (32) + upsampled dec2 (32) = 64
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final output layer to produce single channel image
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder with saving skip connections ---
        e1 = self.enc1(x)          # [B, 32, H, W]
        p1 = self.pool1(e1)        # [B, 32, H/2, W/2]

        e2 = self.enc2(p1)         # [B, 64, H/2, W/2]
        p2 = self.pool2(e2)        # [B, 64, H/4, W/4]

        e3 = self.enc3(p2)         # [B, 128, H/4, W/4]
        p3 = self.pool3(e3)        # [B, 128, H/8, W/8]

        # --- Bottleneck and Attention ---
        b = self.bottleneck(p3)    # [B, 256, H/8, W/8]
        att_map = self.attention(b)# [B, 1, H/8, W/8] Attention map from bottleneck

        # --- Decoder with skip connections ---
        d3_up = self.up3(b)        # Upsample bottleneck features: [B, 128, H/4, W/4]
        # Concatenate with corresponding encoder feature map
        d3_cat = torch.cat([d3_up, e3], dim=1)  # [B, 128+128=256, H/4, W/4]
        d3 = self.dec3(d3_cat)     # Process concatenated features: [B, 128, H/4, W/4]

        d2_up = self.up2(d3)       # Upsample: [B, 64, H/2, W/2]
        d2_cat = torch.cat([d2_up, e2], dim=1)  # Concatenate: [B, 64+64=128, H/2, W/2]
        d2 = self.dec2(d2_cat)     # Process: [B, 64, H/2, W/2]

        d1_up = self.up1(d2)       # Upsample: [B, 32, H, W]
        d1_cat = torch.cat([d1_up, e1], dim=1)  # Concatenate: [B, 32+32=64, H, W]
        d1 = self.dec1(d1_cat)     # Process: [B, 32, H, W]

        # --- Final Reconstruction ---
        # Apply final conv and sigmoid for output in [0, 1]
        reconstruction = torch.sigmoid(self.out_conv(d1)) # [B, 1, H, W]

        return reconstruction, att_map # Return both reconstruction and attention map

# --- 3. DenseNet121 Model Definition for Classification ---

class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(DenseNet121TumorClassificationModel, self).__init__()

        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)

        # Modify first conv layer to accept 1-channel (grayscale) images
        # Keep other parameters like stride, padding the same
        original_conv0 = self.densenet121.features.conv0
        self.densenet121.features.conv0 = nn.Conv2d(1, original_conv0.out_channels,
                                                   kernel_size=original_conv0.kernel_size,
                                                   stride=original_conv0.stride,
                                                   padding=original_conv0.padding,
                                                   bias=False)
        # Optional: Initialize weights for the new conv layer
        if pretrained:
             # Initialize new conv layer weights based on original weights (average across input channels)
             with torch.no_grad():
                 self.densenet121.features.conv0.weight.copy_(original_conv0.weight.mean(dim=1, keepdim=True))


        # Get number of features from the original classifier's input
        num_ftrs = self.densenet121.classifier.in_features

        # Replace the classifier with a new one including dropout
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4), # Increased dropout
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), # Increased dropout
            nn.Linear(256, num_classes)
            # LogSoftmax is often combined with NLLLoss, but CrossEntropyLoss includes it
        )

        # Optional: Initialize weights for the new classifier layers
        self._initialize_weights(self.densenet121.classifier)


    def _initialize_weights(self, classifier_module):
        """Initializes weights for the linear layers in the classifier."""
        for m in classifier_module.modules():
            if isinstance(m, nn.Linear):
                # Kaiming He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize bias to zero
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # DenseNet forward pass (features + classifier)
        # The original DenseNet implementation already includes adaptive pooling
        out = self.densenet121(x)
        return out

# --- 4. Utility Functions ---
def calculate_mean_std(loader):
    """Calculates mean and std dev for normalization based on a DataLoader."""
    mean = 0.
    std = 0.
    total_samples = 0.
    num_batches = 0
    print("Calculating mean and std...")
    for images, _ in tqdm(loader, desc="Mean/Std Calc"):
        # Ensure images are tensors
        if not isinstance(images, torch.Tensor):
            print(f"Warning: Unexpected data type in loader: {type(images)}")
            continue
        # Skip error placeholders if any
        if images.nelement() == 0 or images.shape[1] != 1: # Check for empty or wrong channel tensors
             continue

        batch_samples = images.size(0)
        # Reshape to (batch_size, channels, height * width)
        images = images.view(batch_samples, images.size(1), -1)
        # Calculate mean and std across H*W dimensions, sum over batch
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
        num_batches += 1

    if total_samples == 0:
        print("Warning: No valid samples found to calculate mean/std. Using defaults.")
        return torch.tensor([0.5]), torch.tensor([0.5]) # Default fallback

    mean /= total_samples
    std /= total_samples
    # Ensure std is not zero
    std = torch.max(std, torch.tensor(1e-6))
    return mean, std

# --- 5. Training Loops ---

def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path='enhanced_autoencoder_notumor.pth', attention_weight=0.1):
    """Trains the Enhanced Autoencoder."""
    if os.path.exists(model_path):
        print(f"Loading existing autoencoder from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Autoencoder loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading autoencoder model: {e}. Training from scratch.")

    model = model.to(device)
    # Use MSELoss for reconstruction, as output is sigmoid [0,1] and input is also [0,1]
    criterion_recon = nn.MSELoss()
    # AdamW optimizer is generally good
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) # Adjusted LR and WD

    # Scheduler for learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    print(f"--- Starting Enhanced Autoencoder Training ({num_epochs} epochs) ---")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        recon_loss_epoch = 0.0
        att_loss_epoch = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [AE Train]")
        for images, targets in progress_bar:
            # Ensure targets are the same as images for AE
            images = images.to(device)
            targets = targets.to(device) # Should be same as images

            optimizer.zero_grad()

            # Forward pass: get reconstruction and attention map
            reconstructed, attention_maps = model(images)

            # Calculate reconstruction loss (MSE between input and output)
            loss_recon = criterion_recon(reconstructed, targets)

            # Calculate attention regularization loss (encourage attention map sparsity or focus)
            # Simple L1 penalty on attention map values (mean absolute value)
            loss_att = torch.mean(torch.abs(attention_maps))
            # Alternative: Encourage values closer to 0 or 1 (less common for this purpose)
            # loss_att = torch.mean(- (attention_maps * torch.log(attention_maps + 1e-8) + (1 - attention_maps) * torch.log(1 - attention_maps + 1e-8)))


            # Combined loss
            loss = loss_recon + attention_weight * loss_att

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_loss_epoch += loss_recon.item()
            att_loss_epoch += loss_att.item()

            progress_bar.set_postfix(Loss=f"{loss.item():.4f}", ReconL=f"{loss_recon.item():.4f}", AttL=f"{loss_att.item():.4f}")

        avg_total_loss = total_loss / len(train_loader)
        avg_recon_loss = recon_loss_epoch / len(train_loader)
        avg_att_loss = att_loss_epoch / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Completed:")
        print(f"  Avg Total Loss: {avg_total_loss:.4f}")
        print(f"  Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Avg Atten Loss: {avg_att_loss:.4f}")

        # Step the scheduler based on the validation loss (or training loss if no validation set)
        scheduler.step(avg_total_loss)

        # Save the best model based on total loss
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to {model_path} (Loss: {best_loss:.4f})")

    print("--- Enhanced Autoencoder training complete. ---")
    # Reload the best saved model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best autoencoder model from {model_path}")
    except Exception as e:
         print(f"Warning: Could not reload best autoencoder model after training: {e}")
    return model


def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs=50, device=get_device(),
                    model_path='tumor_classification_densenet121.pth'):
    """Trains the classification model."""

    if os.path.exists(model_path):
        print(f"Loading existing classifier model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Classifier loaded successfully. Skipping training.")
            # Optionally, you could run a validation epoch here to check performance
            return model
        except Exception as e:
            print(f"Error loading classifier model: {e}. Training from scratch.")

    model.to(device)
    best_val_f1 = 0.0 # Track best F1 score for saving model

    print(f"--- Starting Classifier Training ({num_epochs} epochs) ---")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        all_labels_train = []
        all_preds_train = []

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(progress_bar_train):
             # Skip batches with error labels
            if torch.any(labels == -1):
                print(f"Skipping batch {batch_idx} due to loading error.")
                continue
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # Scheduler step (for OneCycleLR, step after each batch)
            if isinstance(scheduler, OneCycleLR):
                 scheduler.step()

            train_loss += loss.item() # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)
            all_labels_train.extend(labels.cpu().numpy())
            all_preds_train.extend(predicted.cpu().numpy())

            progress_bar_train.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{optimizer.param_groups[0]['lr']:.6f}")


        avg_train_loss = train_loss / len(train_loader) # Avg loss per batch
        train_accuracy = 100 * accuracy_score(all_labels_train, all_preds_train)
        train_precision = precision_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_labels_val = []
        all_preds_val = []

        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in progress_bar_val:
                 # Skip batches with error labels
                if torch.any(labels == -1):
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(predicted.cpu().numpy())
                progress_bar_val.set_postfix(Loss=f"{loss.item():.4f}")


        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * accuracy_score(all_labels_val, all_preds_val)
        val_precision = precision_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_accuracy:.2f}%, Val F1:   {val_f1:.4f}")
        print(f"  Val Metrics (Prec/Recall): {val_precision:.4f} / {val_recall:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")


        # --- Learning Rate Scheduling (non-OneCycleLR) ---
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss) # Step based on validation loss

        # --- Save Best Model (based on validation F1 score) ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to {model_path} (Val F1: {best_val_f1:.4f})")

    print("--- Classifier training complete. ---")
     # Reload the best saved model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best classifier model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not reload best classifier model after training: {e}")
    return model

# --- 6. Enhanced Tumor Outlining Function ---

def enhance_outline_detection(reconstruction_error, attention_map, threshold=None):
    """
    Enhanced outline detection using combined reconstruction error and attention maps
    with advanced filtering and morphological refinement.

    Args:
        reconstruction_error (np.ndarray): Pixel-wise absolute difference (normalized or raw).
        attention_map (np.ndarray): Attention map from the autoencoder (normalized or raw).
        threshold (float, optional): Manual threshold for the combined map [0, 1].
                                     If None, Otsu's method is used.

    Returns:
        tuple: (tumor_mask_refined, combined_filtered)
               - tumor_mask_refined (np.ndarray): Binary mask (uint8) of the detected region.
               - combined_filtered (np.ndarray): Filtered combined map (float32, [0, 1]).
    """
    # Normalize the input maps to [0,1] robustly
    rec_min, rec_max = reconstruction_error.min(), reconstruction_error.max()
    att_min, att_max = attention_map.min(), attention_map.max()

    rec_norm = (reconstruction_error - rec_min) / (rec_max - rec_min + 1e-8)
    att_norm = (attention_map - att_min) / (att_max - att_min + 1e-8)

    # Combine the normalized maps (e.g., simple average, or weighted sum)
    # Weighting attention slightly higher might help if it's reliable
    combined_map = (0.4 * rec_norm + 0.6 * att_norm)
    # Ensure combined map is clipped to [0, 1] just in case
    combined_map = np.clip(combined_map, 0, 1)

    # --- Filtering ---
    # Convert combined map to uint8 scale [0, 255] for OpenCV filters
    combined_uint8 = (combined_map * 255).astype(np.uint8)

    # Apply Gaussian blur to smooth out high-frequency noise
    combined_gauss = cv2.GaussianBlur(combined_uint8, (5, 5), 0)

    # Apply bilateral filter: smooths while preserving edges (can be slow)
    # combined_bilat = cv2.bilateralFilter(combined_gauss, d=9, sigmaColor=75, sigmaSpace=75)
    # Using Median filter as a potentially faster alternative for noise removal
    combined_median = cv2.medianBlur(combined_gauss, 5)
    combined_filtered_uint8 = combined_median # Use median filter result

    # Normalize the filtered map back to [0,1] float
    combined_filtered = combined_filtered_uint8.astype(np.float32) / 255.0

    # --- Thresholding ---
    if threshold is None:
        # Use Otsu's method on the *filtered* map
        if combined_filtered_uint8.max() > 0: # Check if the map isn't all black
             otsu_thresh_val = threshold_otsu(combined_filtered_uint8)
             threshold = otsu_thresh_val / 255.0
             print(f"  Otsu threshold determined: {threshold:.4f} (Value: {otsu_thresh_val})")
        else:
             threshold = 0.99 # Default high threshold if map is all black
             print("  Warning: Filtered map is all black, using default high threshold.")
    else:
        print(f"  Using provided threshold: {threshold:.4f}")

    # Create initial binary mask
    tumor_mask = (combined_filtered > threshold).astype(np.uint8)
    # print(f"  Tumor Mask unique values (after threshold): {np.unique(tumor_mask)}")

    # --- Morphological Operations for Refinement ---
    # Define structuring element (ellipse is often good for medical shapes)
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 1. Closing: Fill small holes within the detected regions
    mask_closed = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel_ellipse_5, iterations=2)

    # 2. Opening: Remove small noise/artifacts outside the main regions
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_ellipse_3, iterations=2) # Smaller kernel for opening

    # 3. Optional Dilation: Slightly enlarge the final mask to ensure coverage (use cautiously)
    # tumor_mask_refined = cv2.dilate(mask_opened, kernel_ellipse_3, iterations=1)
    tumor_mask_refined = mask_opened # Use the opened mask directly usually better

    # print(f"  Tumor mask unique values (after refinement): {np.unique(tumor_mask_refined)}")

    return tumor_mask_refined, combined_filtered


def outline_tumor_enhanced(input_image_path, autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                           image_size=(224, 224), error_threshold=None, device=torch.device("cpu"), # Default outlining to CPU for safety
                           show_plots=True): # Added flag to control plotting
    """
    Outlines a potential tumor in an MRI image using a trained enhanced autoencoder
    that incorporates attention mechanisms. Uses preprocessing matching AE training.

    Args:
        input_image_path (str): Path to the input MRI image.
        autoencoder_model_path (str): Path to the trained enhanced autoencoder model.
        image_size (tuple): Target image size (height, width).
        error_threshold (float, optional): Threshold for combined map [0, 1].
                                           If None, Otsu's method is used.
        device (torch.device): Device for autoencoder inference.
        show_plots (bool): Whether to display intermediate and final images using matplotlib.

    Returns:
        tuple: (original_resized_pil, outlined_pil_image)
               - original_resized_pil (PIL.Image): The input image, resized.
               - outlined_pil_image (PIL.Image): Image with tumor outline/overlay, or None if error/no contours.
               Returns (None, None) on major error.
    """
    print(f"\n--- Starting Tumor Outlining for: {os.path.basename(input_image_path)} ---")
    print(f"Using device: {device}")
    try:
        # --- Load Model ---
        loaded_autoencoder = EnhancedAutoencoder()
        if not os.path.exists(autoencoder_model_path):
             print(f"Error: Autoencoder model not found at {autoencoder_model_path}")
             return None, None
        loaded_autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
        loaded_autoencoder.to(device).eval()
        print("Autoencoder model loaded.")

        # --- Load and Prepare Input Image ---
        original_img_pil = Image.open(input_image_path).convert('L')

        # Keep a resized version of the *original* for display and final overlay base
        original_img_resized_pil = original_img_pil.resize(image_size)
        original_img_resized_np = np.array(original_img_resized_pil) # NumPy version for OpenCV

        # --- Preprocessing for Autoencoder Input (Matches AE Training) ---
        # Apply advanced preprocess first to the original size PIL image
        preprocessed_pil = advanced_preprocess(original_img_pil)

        # Define the transform pipeline *without* normalization
        transform_pipeline_ae = transforms.Compose([
            transforms.Resize(image_size),      # Resize *after* advanced_preprocess
            transforms.ToTensor(),              # Converts preprocessed PIL to [0, 1] tensor
            transforms.Lambda(to_float_tensor), # Ensure float
        ])

        # Apply transforms to the preprocessed PIL image and move to device
        img_tensor_ae = transform_pipeline_ae(preprocessed_pil).unsqueeze(0).to(device)
        # print(f"Input tensor shape for AE: {img_tensor_ae.shape}")

        # --- Autoencoder Inference ---
        with torch.no_grad():
            reconstructed_img_tensor, attention_maps_tensor = loaded_autoencoder(img_tensor_ae)

        # --- Process Outputs (Move to CPU and Convert to NumPy) ---
        # Ensure tensors are moved to CPU *before* numpy conversion
        reconstructed_img_np = reconstructed_img_tensor.cpu().squeeze().numpy()
        attention_map_np = attention_maps_tensor.cpu().squeeze().numpy() # Attention map is [H/8, W/8]
        input_img_ae_np = img_tensor_ae.cpu().squeeze().numpy() # This is the preprocessed input to AE

        # print(f"Reconstructed image shape (numpy): {reconstructed_img_np.shape}")
        # print(f"Attention map shape (numpy): {attention_map_np.shape}")

        # --- Calculate Reconstruction Error ---
        # Compare AE input ([0,1]) with AE output ([0,1])
        reconstruction_error = np.abs(input_img_ae_np - reconstructed_img_np)

        # --- Upsample Attention Map ---
        # Upsample the small attention map to the full image size for combination
        attention_map_resized = cv2.resize(attention_map_np, (image_size[1], image_size[0]), # (width, height) for cv2.resize
                                           interpolation=cv2.INTER_LINEAR) # Linear interpolation is usually fine

        # --- Generate Tumor Mask using Enhanced Detection ---
        print("Running enhanced outline detection...")
        tumor_mask, combined_map_filtered = enhance_outline_detection(
            reconstruction_error,
            attention_map_resized,
            threshold=error_threshold # Pass the threshold value
        )
        print(f"Tumor Mask unique values after detection fn: {np.unique(tumor_mask)}") # Should be 0s and 1s

        # --- Find Contours ---
        # Ensure mask is uint8
        tumor_mask_uint8 = tumor_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            tumor_mask_uint8,
            cv2.RETR_EXTERNAL, # Get only outer contours
            cv2.CHAIN_APPROX_SIMPLE # Compress contour points
            # cv2.CHAIN_APPROX_NONE # Use if more detail needed
        )
        print(f"Number of contours found: {len(contours)}")

        # --- Create Outlined Image ---
        # Start with the original resized image converted to BGR for color drawing
        outlined_image_np = cv2.cvtColor(original_img_resized_np, cv2.COLOR_GRAY2BGR)
        overlay = outlined_image_np.copy() # For transparent filling effect
        contours_drawn = 0

        if contours:
            # Filter contours by area
            min_contour_area = 50  # Adjust based on expected tumor size relative to image_size
            max_contour_area = (image_size[0] * image_size[1]) * 0.6 # Ignore very large regions
            valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
            print(f"Number of valid contours after area filtering: {len(valid_contours)}")

            # Sort by area (descending) and take top N (e.g., 3)
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

            for contour in valid_contours:
                # Optional: Further filtering (e.g., aspect ratio, circularity) if needed
                # x, y, w, h = cv2.boundingRect(contour)
                # aspect_ratio = float(w)/h if h > 0 else 0
                # if not (0.2 < aspect_ratio < 5.0): continue # Example filter

                # Draw filled contour on overlay (green, semi-transparent)
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                # Draw contour boundary on main image (red, thicker)
                cv2.drawContours(outlined_image_np, [contour], -1, (255, 0, 0), thickness=2)
                # Optional: Draw bounding box (blue)
                # x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(outlined_image_np, (x, y), (x + w, y + h), (0, 0, 255), 1)
                contours_drawn += 1

            # Blend the overlay with the main image
            alpha = 0.3 # Transparency level
            cv2.addWeighted(overlay, alpha, outlined_image_np, 1 - alpha, 0, outlined_image_np)
            print(f"Drawn {contours_drawn} contours.")


        # --- Plotting (Optional) ---
        if show_plots:
            plt.figure(figsize=(18, 6)) # Wider figure

            plt.subplot(1, 5, 1)
            plt.imshow(original_img_resized_np, cmap='gray')
            plt.title('Original (Resized)')
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.imshow(reconstruction_error, cmap='hot')
            plt.title('Recon Error')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.imshow(attention_map_resized, cmap='viridis')
            plt.title('Attention Map')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.subplot(1, 5, 4)
            plt.imshow(tumor_mask_uint8, cmap='gray')
            plt.title('Final Mask')
            plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.imshow(cv2.cvtColor(outlined_image_np, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for plt
            plt.title('Outlined Result')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        # --- Return Results ---
        outlined_pil_image = Image.fromarray(outlined_image_np) if contours_drawn > 0 else original_img_resized_pil
        # Return original resized PIL and the outlined version as PIL
        return original_img_resized_pil, outlined_pil_image

    except Exception as e:
        print(f"Error during enhanced tumor outlining: {e}")
        print(traceback.format_exc()) # Print detailed traceback
        return None, None # Return None for both images on error


# --- 7. Prediction and Outlining ---

# Import accuracy_score here as it's used in training loops now
from sklearn.metrics import accuracy_score

def predict_and_outline_simple(input_image_path,
                            classification_model_path='tumor_classification_densenet121.pth',
                            autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                            image_size=(224, 224),
                            error_threshold=None, # Threshold for outlining
                            normalize_transform=None, # Pass the normalization transform
                            device_classifier=get_device(),
                            device_outlining=None, # Can be different (e.g., CPU)
                            show_plots=True): # Control plotting
    """
    Predicts tumor type using DenseNet and outlines using Enhanced Autoencoder.

    Args:
        input_image_path (str): Path to the input MRI image.
        classification_model_path (str): Path to the trained classification model.
        autoencoder_model_path (str): Path to the trained enhanced autoencoder model.
        image_size (tuple): Target image size for models.
        error_threshold (float, optional): Threshold for outlining's combined map [0, 1]. Otsu if None.
        normalize_transform (transforms.Normalize, optional): The normalization transform used for the classifier.
        device_classifier (torch.device): Device for classification model.
        device_outlining (torch.device, optional): Device for outlining model. Defaults to CPU if None.
        show_plots (bool): Whether to display the final comparison plot.

    Returns:
        tuple: (predicted_class_name, confidence) or (None, None) on error.
    """
    print(f"\n--- Starting Prediction & Outlining for: {os.path.basename(input_image_path)} ---")

    if device_outlining is None:
        device_outlining = torch.device("cpu") # Default outlining to CPU for stability if not specified
        print("Outlining device not specified, defaulting to CPU.")

    if normalize_transform is None:
         print("Warning: Normalization transform not provided for classifier. Using default [0.5, 0.5].")
         normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    try:
        # --- Load Classification Model ---
        loaded_classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False) # Pretrained=False if loading state_dict
        if not os.path.exists(classification_model_path):
            print(f"Error: Classifier model not found at {classification_model_path}")
            return None, None
        loaded_classifier.load_state_dict(torch.load(classification_model_path, map_location=device_classifier))
        loaded_classifier.to(device_classifier).eval()
        class_map = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'} # Match MRIDataset map
        print("Classifier model loaded.")

        # --- Prepare Image for Classifier ---
        original_img_pil = Image.open(input_image_path).convert('L')

        # Preprocessing pipeline *for the classifier* (includes normalization)
        transform_pipeline_classify = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess), # Apply same initial preprocessing
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
            normalize_transform # Apply the specific normalization for classifier
        ])
        img_tensor_classify = transform_pipeline_classify(original_img_pil).unsqueeze(0).to(device_classifier)

        # --- Classification Prediction ---
        with torch.no_grad():
            prediction = loaded_classifier(img_tensor_classify)
            probabilities = torch.softmax(prediction, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)
            predicted_class_index = predicted_class_index.item()
            confidence = confidence.item()

        predicted_class_name = class_map.get(predicted_class_index, "Unknown")
        print(f"Predicted Tumor Type: {predicted_class_name} (Confidence: {confidence:.4f})")

        # --- Tumor Outlining (if tumor predicted) ---
        original_resized_pil = None
        outlined_final_image_pil = None

        if predicted_class_name != 'notumor':
            print("Tumor class predicted, attempting outlining...")
            # Call the enhanced outlining function
            # It returns (original_resized_pil, outlined_pil_image_or_original)
            original_resized_pil, outlined_final_image_pil = outline_tumor_enhanced(
                input_image_path=input_image_path,
                autoencoder_model_path=autoencoder_model_path,
                image_size=image_size,
                error_threshold=error_threshold,
                device=device_outlining, # Use specified device for outlining
                show_plots=False # Control intermediate plots from here
            )
            if outlined_final_image_pil is None:
                 print("Outlining function returned an error.")
                 # Fallback: use original resized image for display
                 original_resized_pil = original_img_pil.resize(image_size)
                 outlined_final_image_pil = original_resized_pil
            elif original_resized_pil is None: # Should not happen if outlining didn't error, but check
                 print("Error retrieving original resized image from outlining.")
                 return predicted_class_name, confidence # Return prediction even if plotting fails

        else:
            print("No tumor class predicted.")
            # If no tumor, the 'outlined' image is just the original resized image
            original_resized_pil = original_img_pil.resize(image_size)
            outlined_final_image_pil = original_resized_pil


        # --- Display Final Result ---
        if show_plots and original_resized_pil and outlined_final_image_pil:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(original_resized_pil, cmap='gray')
            plt.title('Original MRI (Resized)')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            # Convert PIL to NumPy for displaying with matplotlib
            plt.imshow(np.array(outlined_final_image_pil)) # Assumes outlined is BGR if color, handled by PIL->np conversion
            if predicted_class_name != 'notumor':
                 if outlined_final_image_pil != original_resized_pil: # Check if outlining actually happened
                     title = f'{predicted_class_name.capitalize()} Detected (Outlined)\nConfidence: {confidence:.2%}'
                 else:
                     title = f'{predicted_class_name.capitalize()} Detected (Outline Failed)\nConfidence: {confidence:.2%}'
            else:
                 title = f'No Tumor Detected\nConfidence: {confidence:.2%}'
            plt.title(title)
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        return predicted_class_name, confidence

    except Exception as e:
        print(f"An error occurred during prediction and outlining: {e}")
        print(traceback.format_exc())
        return None, None

# --- 8. Main Execution ---

def main():
    freeze_support() # Necessary for multiprocessing esp. on Windows

    # --- Configuration ---
    device = get_device()
    # data_dir = 'path/to/your/dataset/Training' # <<< --- CHANGE THIS TO YOUR DATASET PATH
    data_dir = 'E:\\project\\dataset1\\training1' # Example path
    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        print("Please update the 'data_dir' variable in the main() function.")
        return

    image_size = (256, 256) # Standard size for many pretrained models like DenseNet
    batch_size = 32         # Adjust based on GPU/XPU memory
    num_epochs_classifier = 25 # Number of epochs for classifier training
    num_epochs_autoencoder = 20 # Number of epochs for autoencoder training
    num_workers = 4 # Adjust based on your CPU cores and system
    classifier_model_path = 'tumor_classification_densenet121_best.pth'
    autoencoder_model_path = 'enhanced_autoencoder_notumor_best.pth'

    # --- Data Loading and Preparation ---
    print("\n--- Loading Datasets ---")
    # Load full dataset once for splitting and calculating stats (don't apply normalization yet)
    full_dataset_for_stats = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=False)

    if len(full_dataset_for_stats) == 0:
        print("Error: No images found in the dataset. Exiting.")
        return

    # Split indices
    try:
        train_idx, temp_idx = train_test_split(
            list(range(len(full_dataset_for_stats))),
            test_size=0.3, # 70% train, 30% temp
            random_state=42,
            stratify=full_dataset_for_stats.labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5, # Split temp 50/50 -> 15% val, 15% test
            random_state=42,
            stratify=[full_dataset_for_stats.labels[i] for i in temp_idx]
        )
    except ValueError as e:
         print(f"Error during train/val/test split, likely due to small class sizes: {e}")
         print("Consider using a larger dataset or adjusting split ratios.")
         return

    # Create a temporary loader *without normalization* to calculate mean/std
    temp_train_dataset = torch.utils.data.Subset(full_dataset_for_stats, train_idx)
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_mean, train_std = calculate_mean_std(temp_train_loader)
    print(f"Calculated Mean: {train_mean.item():.4f}, Std: {train_std.item():.4f} from training set")

    # Define the normalization transform based on calculated stats
    global normalize_transform # Make accessible to prediction function if needed elsewhere
    normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)

    # Now create the final datasets and apply normalization to the classification dataset
    # Re-create the full dataset instance, this time enabling normalization
    full_classification_dataset = MRIDataset(data_dir, image_size=image_size, split='full', apply_normalize=True)
    full_classification_dataset.set_normalization(normalize_transform) # Set the calculated normalization

    # Create subsets using the *new* dataset instance which has normalization configured
    train_classification_dataset = torch.utils.data.Subset(full_classification_dataset, train_idx)
    val_classification_dataset = torch.utils.data.Subset(full_classification_dataset, val_idx)
    test_classification_dataset = torch.utils.data.Subset(full_classification_dataset, test_idx) # Keep test set aside

    # Create DataLoaders for Classifier Training
    train_classification_loader = DataLoader(
        train_classification_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_classification_loader = DataLoader(
        val_classification_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    # Test loader (not used in training loop, but good practice to have)
    test_classification_loader = DataLoader(
        test_classification_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    # Create Dataset and DataLoader for Autoencoder Training ('notumor' images only)
    notumor_autoencoder_dataset = NotumorMRIDataset(data_dir, image_size=image_size, split='train')
    if len(notumor_autoencoder_dataset) == 0:
        print("Warning: No 'notumor' images found for autoencoder training.")
        # Decide how to handle this: skip AE training, use different data, etc.
        # For now, we might skip AE training if the loader is empty.
        train_autoencoder_loader = None
    else:
        train_autoencoder_loader = DataLoader(
            notumor_autoencoder_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
        )

    # --- Autoencoder Training ---
    if train_autoencoder_loader:
        print("\n--- Training Enhanced Autoencoder ---")
        autoencoder = EnhancedAutoencoder()
        trained_autoencoder = train_enhanced_autoencoder(
            autoencoder,
            train_autoencoder_loader,
            num_epochs=num_epochs_autoencoder,
            device=device,
            model_path=autoencoder_model_path,
            attention_weight=0.05 # Weight for the attention sparsity loss
        )
    else:
        print("\n--- Skipping Autoencoder Training (No Data) ---")
        # Attempt to load if exists, otherwise outlining might fail later
        if os.path.exists(autoencoder_model_path):
             print(f"Attempting to load existing autoencoder from {autoencoder_model_path}")
             # We don't need the trained_autoencoder object here unless we evaluate it
        else:
             print(f"Warning: Autoencoder model not found at {autoencoder_model_path} and cannot be trained.")


    # --- Classifier Training ---
    print("\n--- Training Classifier ---")
    classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=True)

    # Calculate class weights for addressing imbalance
    class_counts = Counter(full_classification_dataset.labels)
    if len(class_counts) < 4:
        print("Warning: Not all classes found in dataset labels. Weight calculation might be inaccurate.")
        # Use uniform weights as fallback or handle appropriately
        class_weights = torch.ones(4, dtype=torch.float)
    else:
        # Ensure counts are in the correct order (0, 1, 2, 3)
        counts = np.array([class_counts[i] for i in range(4)])
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / np.sum(weights) # Normalize weights
        class_weights = torch.tensor(weights * len(class_counts), dtype=torch.float).to(device) # Scale weights

    print(f"Calculated Class Weights: {class_weights.cpu().numpy()}")
    criterion_classifier = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (AdamW is generally robust)
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4) # Lower LR for fine-tuning

    # Scheduler (OneCycleLR often works well)
    total_steps = num_epochs_classifier * len(train_classification_loader)
    scheduler_classifier = OneCycleLR(
        optimizer_classifier,
        max_lr=1e-3, # Max LR during cycle
        total_steps=total_steps,
        pct_start=0.2, # Warm-up phase duration
        anneal_strategy='cos', # Cosine annealing
        div_factor=10,    # Initial LR = max_lr / div_factor
        final_div_factor=1e4 # Final LR = Initial LR / final_div_factor
    )
    # Alternatively, use ReduceLROnPlateau:
    # scheduler_classifier = ReduceLROnPlateau(optimizer_classifier, mode='max', factor=0.2, patience=5, verbose=True) # Step based on Val F1

    trained_classifier = train_classifier(
        classifier,
        train_classification_loader,
        val_classification_loader,
        criterion_classifier,
        optimizer_classifier,
        scheduler=scheduler_classifier, # Pass the scheduler
        num_epochs=num_epochs_classifier,
        device=device,
        model_path=classifier_model_path
    )

    # --- Prediction and Outlining on a Test Image ---
    print("\n--- Performing Prediction and Outlining on a Sample Image ---")
    # Select a sample image (replace with a path from your test set or any image)
    # sample_image_path = "path/to/your/test/image.jpg" # <<< --- CHANGE THIS
    sample_image_path = "E:\\project\\dataset1\\training1\\meningioma\\Te-me_0085.jpg" # Example path

    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image for prediction not found at: {sample_image_path}")
        # Try to find one from the test set if available
        if test_idx and hasattr(full_classification_dataset, 'image_paths'):
             try:
                 sample_image_path = full_classification_dataset.image_paths[test_idx[0]]
                 print(f"Using sample image from test set: {sample_image_path}")
             except IndexError:
                 print("Could not get a sample image from the test set.")
                 sample_image_path = None
        else:
            sample_image_path = None

    if sample_image_path:
        # Perform prediction and outlining
        predict_and_outline_simple(
            input_image_path=sample_image_path,
            classification_model_path=classifier_model_path,
            autoencoder_model_path=autoencoder_model_path,
            image_size=image_size,
            error_threshold=None, # Use Otsu thresholding by default
            normalize_transform=normalize_transform, # Pass the calculated normalization
            device_classifier=device,
            device_outlining=torch.device("cpu"), # Run outlining on CPU for potentially more stability/memory
            show_plots=True # Show the final plot
        )
    else:
        print("Skipping final prediction demo as no sample image path was found/valid.")

    print("\n--- Script Finished ---")


if __name__ == '__main__':
    freeze_support() 
    main()