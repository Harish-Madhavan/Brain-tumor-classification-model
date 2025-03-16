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
# --- 1. Data Loading and Preprocessing ---

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

class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train'):
        self.image_paths = []
        self.labels = []
        class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_labels_map = {folder: i for i, folder in enumerate(class_folders)}
        self.class_counts = {folder: 0 for folder in class_folders}

        print(f"Loading {split} data from directory: {data_dir}")

        for class_folder in class_folders:
            class_dir = os.path.join(data_dir, class_folder)
            print(f"  Processing folder: {class_folder}")
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    image_path = os.path.join(class_dir, filename)
                    try:
                        self.image_paths.append(image_path)
                        self.labels.append(self.class_labels_map[class_folder])
                        self.class_counts[class_folder] += 1
                    except Exception as e:
                        print(f"    Error listing image: {filename}, error: {e}")
        self.image_size = image_size
        self.split = split

        self.base_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(lambda img: advanced_preprocess(img)),  # New advanced pre-processing
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.float()),
        ])

        # --- Expanded Data Augmentation ---
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),  # Increased rotation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  # Increased translation
                scale=(0.85, 1.15),     # Increased scaling range
                shear=15                 # Increased shear
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # Added saturation and hue
            transforms.RandomGrayscale(p=0.1),   # Randomly convert to grayscale (sometimes helpful)
        ])

        print(f"{split} data loading complete. Found {len(self.image_paths)} images.")
        print(f"Class Distribution for {split} set: {self.class_counts}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = self.base_transforms(image_pil)
            if self.split == 'train':
                image = self.augment_transforms(image)
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading or processing image: {image_path}, error: {e}")
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.tensor(0, dtype=torch.long)

class NotumorMRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train'):
        self.image_paths = []
        notumor_folder = 'notumor'
        print(f"Loading {split} 'notumor' data from directory: {data_dir}")
        notumor_dir = os.path.join(data_dir, notumor_folder)
        print(f"  Processing folder: {notumor_folder}")

        for filename in os.listdir(notumor_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(notumor_dir, filename)
                try:
                    self.image_paths.append(image_path)
                except Exception as e:
                    print(f"    Error listing image: {filename}, error: {e}")

        self.image_size = image_size
        self.split = split
        self.base_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(lambda img: advanced_preprocess(img)),  # New advanced pre-processing
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.float()),
        ])
        print(f"{split} 'notumor' data loading complete. Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = self.base_transforms(image_pil)
            return image, image
        except Exception as e:
            print(f"Error loading or processing image: {image_path}, error: {e}")
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))

# --- 2. Simplified Autoencoder Model ---

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_pool2(x)
        x = F.relu(self.dec_conv1(x))
        x = torch.sigmoid(self.dec_conv2(x))
        return x

class EnhancedAutoencoder(nn.Module):
    def __init__(self):
        super(EnhancedAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bottleneck
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # First block
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Second block
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Third block
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final output
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        features = self.encoder(x)

        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Decode
        decoded = self.decoder(attended_features)
        return decoded, attention_weights

# --- 3. DenseNet121 Model Definition for Classification ---

class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(DenseNet121TumorClassificationModel, self).__init__()

        # Load pretrained DenseNet121
        self.densenet121 = models.densenet121(pretrained=pretrained)

        # Modify first conv layer to accept grayscale images
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                                   padding=3, bias=False)

        # Get number of features from last layer
        num_ftrs = self.densenet121.classifier.in_features

        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        # Create classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.densenet121.classifier = self.classifier

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

# --- 4. Data Loading and Preparation ---

# Define device_xpu
device_xpu = torch.device("xpu" if torch.xpu.is_available() else "cpu")

data_dir = 'E:\\project\\dataset1\\training1'  # Replace with your dataset path
image_size = (256, 256)
batch_size = 32

full_classification_dataset = MRIDataset(data_dir, image_size=image_size, split='full')
train_idx, temp_idx = train_test_split(list(range(len(full_classification_dataset))), test_size=0.3, random_state=42, stratify=full_classification_dataset.labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[full_classification_dataset.labels[i] for i in temp_idx])

train_classification_dataset = torch.utils.data.Subset(full_classification_dataset, train_idx)
val_classification_dataset = torch.utils.data.Subset(full_classification_dataset, val_idx)
test_classification_dataset = torch.utils.data.Subset(full_classification_dataset, test_idx)

def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_samples = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    mean /= total_samples
    std /= total_samples
    return mean, std

temp_train_classification_loader = DataLoader(train_classification_dataset, batch_size=batch_size, shuffle=False)
train_mean, train_std = calculate_mean_std(temp_train_classification_loader)
print(f"Calculated mean: {train_mean}, std: {train_std} from training set")

normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)
full_classification_dataset.base_transforms.transforms.append(normalize_transform)

train_classification_dataset = torch.utils.data.Subset(full_classification_dataset, train_idx)
val_classification_dataset = torch.utils.data.Subset(full_classification_dataset, val_idx)
test_classification_dataset = torch.utils.data.Subset(full_classification_dataset, test_idx)

train_classification_loader = DataLoader(
    train_classification_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,           # Increased number of workers to improve throughput
    pin_memory=True,         # Enables faster host-to-device transfer (even on xpu/CPU setups)
    persistent_workers=True  # Keeps workers alive across epochs, reducing startup overhead
)

val_classification_loader = DataLoader(
    val_classification_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_classification_loader = DataLoader(
    test_classification_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

notumor_autoencoder_dataset = NotumorMRIDataset(data_dir, image_size=image_size, split='train')
train_autoencoder_loader = DataLoader(
    notumor_autoencoder_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# --- 5. Training (Enhanced Autoencoder) ---
def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path):
    if os.path.exists(model_path):
        print(f"Loading autoencoder from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Reset every 5 epochs
        T_mult=2,  # Double the reset interval after each reset
        eta_min=1e-6
    )

    print("Training Enhanced Autoencoder:")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        reconstruction_loss = 0.0
        attention_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            optimizer.zero_grad()

            reconstructed, attention_maps = model(images)

            # Reconstruction loss
            recon_loss = criterion(reconstructed, images)

            # Attention regularization (encourage sparsity)
            att_loss = torch.mean(attention_maps)

            # Combined loss
            loss = recon_loss + 0.1 * att_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            reconstruction_loss += recon_loss.item()
            attention_loss += att_loss.item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_recon_loss = reconstruction_loss / len(train_loader)
        avg_att_loss = attention_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Total Loss: {avg_train_loss:.4f}")
        print(f"Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"Attention Loss: {avg_att_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), model_path)
    print("Enhanced Autoencoder training complete.")
    return model

# Initialize and train the enhanced autoencoder
autoencoder = EnhancedAutoencoder().to(device_xpu)
autoencoder = train_enhanced_autoencoder(
    autoencoder,
    train_autoencoder_loader,
    num_epochs=25,
    device=device_xpu,
    model_path='enhanced_autoencoder_notumor.pth'
)

# --- Training Loop for DenseNet121 Classifier ---

def train_classifier(model, train_loader, val_loader, criterion, optimizer,
                    num_epochs=50, device=device_xpu,
                    model_path='tumor_classification_densenet121.pth',
                    patience=7, scheduler=None):
    if os.path.exists(model_path):
        print("Loading the saved model")
        model.load_state_dict(torch.load(model_path))
        return model

    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0  # For early stopping

    # --- Learning Rate Scheduler ---
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        all_labels = []
        all_preds = []

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_labels_val = []
        all_preds_val = []


        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(predicted.cpu().numpy())


        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = (np.array(all_preds_val) == np.array(all_labels_val)).mean() * 100
        val_precision = precision_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)



        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}, Train F1: {train_f1:.2f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val Precision: {val_precision:.2f}, Val Recall: {val_recall:.2f}, Val F1: {val_f1:.2f}')

        # --- Learning Rate Scheduling ---
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # --- Save Best Model and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {patience} epochs with no improvement.')
                break  # Stop training

    return model

# --- 6. Enhanced Tumor Outlining Function ---
def outline_tumor_enhanced(input_image_path, autoencoder_model_path='enhanced_autoencoder_notumor.pth',
                        image_size=(224, 224), error_threshold=None, device=torch.device("xpu")):
    """
    Outlines a potential tumor in an MRI image using a trained enhanced autoencoder
    that incorporates attention mechanisms.

    Args:
        input_image_path (str): Path to the input MRI image.
        autoencoder_model_path (str): Path to the trained enhanced autoencoder model.
        image_size (tuple): Desired image size (height, width).
        error_threshold (float, optional): Threshold for reconstruction error/combined map.
            If None, it's dynamically calculated using Otsu's method.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: (original_image, outlined_image), both PIL Images.
               Returns (None, None) on error.
    """
    try:
        loaded_autoencoder = EnhancedAutoencoder() # Load EnhancedAutoencoder
        loaded_autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
        loaded_autoencoder.to(device).eval()

        original_img_pil = Image.open(input_image_path).convert('L')
        img = original_img_pil.resize(image_size)
        original_img_np = np.array(img)

        transform_pipeline = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.float()),
            # Assuming normalize_transform is defined globally (as in the original code)
            normalize_transform if 'normalize_transform' in globals() else transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img_tensor = transform_pipeline(img).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed_img_tensor, attention_maps = loaded_autoencoder(img_tensor) # Get both outputs
            reconstructed_img_np = reconstructed_img_tensor.cpu().squeeze().numpy()
            attention_map_np = attention_maps.cpu().squeeze().numpy()

        input_img_np = img_tensor.cpu().squeeze().numpy()
        reconstruction_error = np.abs(input_img_np - reconstructed_img_np)

        # Upsample attention map to the size of reconstruction error
        attention_map_resized = cv2.resize(attention_map_np, image_size, interpolation=cv2.INTER_LINEAR)

        # Use enhanced outline detection function with resized attention map
        tumor_mask, combined_map = enhance_outline_detection(reconstruction_error, attention_map_resized, threshold=error_threshold)

        print(f"Otsu Threshold used: {error_threshold if error_threshold is not None else threshold_otsu(combined_map)}") # Print threshold value
        print(f"Tumor Mask unique values: {np.unique(tumor_mask)}") # Check if mask is binary

        # Find contours with hierarchy - same as before
        contours, hierarchy = cv2.findContours(
            (tumor_mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        print(f"Number of contours found: {len(contours)}") # Print contour count

        # Create output image - same as before
        outlined_image_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2BGR)
        valid_contours = [] # Initialize here so it's defined even if no contours

        if contours:
            # Relax contour area filtering for debugging
            min_contour_area = 20  # Reduced from 100, original was 20
            max_contour_area = (image_size[0] * image_size[1]) * 0.5  # Increased max area, or you can even comment out max constraint for now
            valid_contours = [cnt for cnt in contours
                            if min_contour_area < cv2.contourArea(cnt) ] # Removed max_contour_area constraint for now, you can add it back later for fine-tuning
            print(f"Number of valid contours after area filtering: {len(valid_contours)}") # Print valid contour count

            # Sort contours by area - same as before
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            # Create overlay for transparency - same as before
            overlay = outlined_image_np.copy()

            # Draw only top 3 largest valid contours - same as before
            for i, contour in enumerate(valid_contours[:3]):
                # Calculate contour area - same as before
                area = cv2.contourArea(contour)

                # Calculate bounding box - same as before
                x, y, w, h = cv2.boundingRect(contour)

                # Additional filtering: aspect ratio and size constraints - same as before
                aspect_ratio = float(w)/h
                if not (0.2 < aspect_ratio < 5):  # Filter out extremely elongated contours
                    continue

                # Draw filled contour with reduced transparency - same as before
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)

                # Draw contour boundary with increased thickness - same as before
                cv2.drawContours(outlined_image_np, [contour], -1, (255, 0, 0), 3)

                # Add bounding box - same as before
                cv2.rectangle(outlined_image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Apply transparency with reduced alpha for the overlay - same as before
            cv2.addWeighted(overlay, 0.2, outlined_image_np, 0.8, 0, outlined_image_np)

        plt.figure(figsize=(15, 10)) # Adjusted figure size for better visualization
        plt.subplot(2, 2, 1) # 2x2 subplot grid
        plt.imshow(original_img_np, cmap='gray')
        plt.title('Original')

        plt.subplot(2, 2, 2) # 2x2 subplot grid
        plt.imshow(reconstruction_error, cmap='hot')
        plt.title('Reconstruction Error')
        plt.colorbar()

        plt.subplot(2, 2, 3) # 2x2 subplot grid
        plt.imshow(attention_map_resized, cmap='viridis') # Using viridis colormap for attention map - now resized
        plt.title('Attention Map')
        plt.colorbar()

        plt.subplot(2, 2, 4) # 2x2 subplot grid
        plt.imshow(tumor_mask, cmap='gray')
        plt.title('Tumor Mask')
        plt.show()
        return Image.fromarray(original_img_np), Image.fromarray(outlined_image_np)
    except Exception as e:
        print(f"Error in enhanced tumor outlining: {e}")
        return None, None

def enhance_outline_detection(reconstruction_error, attention_map, threshold=None):
    """Enhanced outline detection using both reconstruction error and attention maps"""
    # Normalize both maps
    reconstruction_error_normalized = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
    attention_map_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    print(f"Reconstruction Error range after normalization: min={reconstruction_error_normalized.min():.4f}, max={reconstruction_error_normalized.max():.4f}")
    print(f"Attention Map range after normalization: min={attention_map_normalized.min():.4f}, max={attention_map_normalized.max():.4f}")

    # Combine the maps
    combined_map = (reconstruction_error_normalized + attention_map_normalized) / 2

    print(f"Combined Map range: min={combined_map.min():.4f}, max={combined_map.max():.4f}")
    plt.figure(figsize=(6, 6))
    plt.imshow(combined_map, cmap='gray')
    plt.title('Combined Map (Before Threshold)')
    plt.colorbar()
    plt.show()

    # Determine threshold using Otsu's method if not provided
    if threshold is None:
        threshold = threshold_otsu(combined_map)
    print(f"Threshold value in enhance_outline_detection: {threshold:.4f}")

    # Create binary mask
    tumor_mask = (combined_map > threshold).astype(np.uint8)
    print(f"Tumor Mask unique values before morphology: {np.unique(tumor_mask)}")

    # --- Refined Morphological Processing ---
    # Define kernels with adjusted sizes:
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small false positives (noise)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel_open)
    # Fill small holes within the tumor region
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel_close)
    # Dilate to merge fragmented tumor regions (helps reduce false skull outlines)
    tumor_mask = cv2.dilate(tumor_mask, kernel_dilate, iterations=1)
    
    print(f"Tumor Mask unique values after refined morphology: {np.unique(tumor_mask)}")
    return tumor_mask, combined_map

# --- 7. Prediction and Outlining (Classification on XPU, Outlining on CPU) ---

def predict_and_outline_simple(input_image_path, classification_model_path='tumor_classification_densenet121.pth', autoencoder_model_path='enhanced_autoencoder_notumor.pth', image_size=(224, 224), error_threshold=None, device_classifier=torch.device("xpu"), device_outlining=None):
  """
    Predicts the tumor type in an MRI image and outlines the tumor (if present) using Enhanced Autoencoder.

    Args:
        input_image_path (str): Path to the input MRI image.
        classification_model_path (str): Path to the trained classification model.
        autoencoder_model_path (str): Path to the trained enhanced autoencoder model.
        image_size (tuple): Desired image size (height, width).
        error_threshold (float, optional): Threshold for reconstruction error/combined map.
            If None, it's dynamically calculated.
        device_classifier (torch.device): Device for classification.
        device_outlining (torch.device, optional): Device for outlining. Defaults to device_classifier if None.

    Returns:
        tuple: (predicted_class_name, confidence)
        Returns (None, None) if any error occurs.
    """
  if device_outlining is None: # Default outlining device to classifier device
      device_outlining = device_classifier

  try:
      loaded_classifier = DenseNet121TumorClassificationModel(num_classes=4)
      loaded_classifier.load_state_dict(torch.load(classification_model_path, map_location=device_classifier))
      loaded_classifier.to(device_classifier).eval()
      class_folders = ['glioma', 'meningioma', 'pituitary', 'notumor']

      original_img_pil = Image.open(input_image_path).convert('L')
      img = original_img_pil.resize(image_size)

      transform_pipeline_classify = transforms.Compose([
          transforms.Resize(image_size),
          transforms.ToTensor(),
          transforms.Lambda(lambda img: img.float()),
          normalize_transform if 'normalize_transform' in globals() else transforms.Normalize(mean=[0.5], std=[0.5])
      ])
      img_tensor = transform_pipeline_classify(img).unsqueeze(0).to(device_classifier)

      with torch.no_grad():
          prediction = loaded_classifier(img_tensor)
          probabilities = torch.softmax(prediction, dim=1)
          predicted_class_index = torch.argmax(probabilities, dim=1).item()
          confidence = probabilities[0, predicted_class_index].item()
      predicted_class_name = class_folders[predicted_class_index]
      print(f"Predicted Tumor Type: {predicted_class_name} (Confidence: {confidence:.4f})")

      original_resized_img = original_img_pil.resize(image_size)
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.imshow(original_resized_img, cmap='gray')
      plt.title('Original MRI')

      if predicted_class_name != 'notumor':
          original_img_outlined, masked_image = outline_tumor_enhanced( # Use enhanced outlining function
              input_image_path,
              autoencoder_model_path='enhanced_autoencoder_notumor.pth',
              image_size= image_size,
              error_threshold=error_threshold, # Pass error_threshold from predict function
              device=device_outlining
            )

          if original_img_outlined is not None and masked_image is not None:
              plt.subplot(1, 2, 2)
              plt.imshow(np.array(masked_image))
              plt.title(f'{predicted_class_name.capitalize()} Tumor Detected\nConfidence: {confidence:.2%}')
              plt.axis('off')
          else:
              plt.subplot(1, 2, 2)
              plt.imshow(original_resized_img, cmap='gray')
              plt.title('Tumor outlining failed')
              plt.axis('off')
      else:
          plt.subplot(1, 2, 2)
          plt.imshow(original_resized_img, cmap='gray')
          plt.title('No Tumor Detected\nConfidence: {:.2%}'.format(confidence))
          plt.axis('off')

      plt.tight_layout()
      plt.show()
      return predicted_class_name, confidence
  except Exception as e:
      print(f"An error occurred during prediction and outlining: {e}")
      return None, None

# --- Train and Use the System ---

# 1. Train the Autoencoder (if not already trained)

# 2. Train the DenseNet Classifier
# --- Training Configuration ---
num_epochs = 50
classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=True)

# Calculate class weights to handle imbalanced data
class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_counts = np.array([full_classification_dataset.class_counts[c] for c in class_folders])
class_weights = torch.tensor((1.0 / class_counts) * (len(full_classification_dataset) / 4),
                           dtype=torch.float).to(device_xpu)

# Use weighted loss
criterion_classifier = nn.CrossEntropyLoss(weight=class_weights)

# Use AdamW optimizer with weight decay
optimizer_classifier = torch.optim.AdamW(classifier.parameters(),
                                       lr=0.001,
                                       weight_decay=0.01,
                                       betas=(0.9, 0.999))

# Learning rate scheduler with warmup
from torch.optim.lr_scheduler import OneCycleLR

# Calculate total steps for OneCycleLR
total_steps = num_epochs * len(train_classification_loader)

scheduler = OneCycleLR(
    optimizer_classifier,
    max_lr=0.001,
    total_steps=total_steps,
    pct_start=0.3,  # 30% of training for warmup
    div_factor=25,  # initial_lr = max_lr/25
    final_div_factor=1e4  # final_lr = initial_lr/1e4
)

# Update training parameters
trained_classifier = train_classifier(
    classifier,
    train_classification_loader,
    val_classification_loader,
    criterion_classifier,
    optimizer_classifier,
    num_epochs=num_epochs,
    device=device_xpu,
    patience=7,  # Increased patience
    scheduler=scheduler  # Pass scheduler to training function
)

# 3. Predict and Outline
def predict_and_outline_simple_fixed_call():
    input_mri_path = "E:\\project\\dataset1\\Testing\\meningioma\\Te-me_0017.jpg"
    predicted_type, confidence = predict_and_outline_simple(
        input_image_path=input_mri_path, # Changed to keyword argument explicitly
        classification_model_path='tumor_classification_densenet121.pth',
        autoencoder_model_path='enhanced_autoencoder_notumor.pth', # Use enhanced autoencoder model path
        error_threshold=None,  # Let Otsu decide threshold for now
        device_classifier=device_xpu,
        device_outlining=device_xpu # Use device_xpu for outlining as well if available
    )
    print(f"Predicted type: {predicted_type}, Confidence: {confidence}")

predict_and_outline_simple_fixed_call()