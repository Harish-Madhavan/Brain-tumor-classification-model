# --- IMPORTS ---
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from skimage.filters import threshold_otsu
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from multiprocessing import freeze_support
import traceback # For detailed error printing
from pytorch_msssim import ssim, MS_SSIM
import seaborn as sns
import pandas as pd # For Excel logging

# --- CBAM Components ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__(); self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1); self.shared_mlp = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x)); max_out = self.shared_mlp(self.max_pool(x)); channel_att = self.sigmoid(avg_out + max_out); return x * channel_att
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__(); assert kernel_size % 2 == 1, "Kernel size must be odd"; padding = kernel_size // 2; self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True); pooled_features = torch.cat([avg_out, max_out], dim=1); spatial_att = self.sigmoid(self.conv(pooled_features)); return x * spatial_att
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__(); self.channel_attention = ChannelAttention(in_channels, reduction_ratio); self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x); x = self.spatial_attention(x); return x

# --- Helper Functions ---
def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available(): print("Using XPU device."); return torch.device("xpu")
    elif torch.cuda.is_available(): print("XPU not available, using CUDA device."); return torch.device("cuda")
    else: print("XPU and CUDA not available, using CPU device."); return torch.device("cpu")
def equalize_histogram(img): return ImageOps.equalize(img)
def advanced_preprocess(img):
    img_np = np.array(img); clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); img_clahe = clahe.apply(img_np); img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0); return Image.fromarray(img_blur)
def to_float_tensor(img): return img.float()

# --- Data Loading and Preprocessing ---
class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', apply_normalize=True):
        self.image_paths, self.labels = [], []; class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']; self.class_labels_map = {folder: i for i, folder in enumerate(class_folders)}; self.class_counts = {folder: 0 for folder in class_folders}; self.apply_normalize = apply_normalize
        # print(f"Loading {split} data from directory: {data_dir}")
        for class_folder in class_folders:
            class_dir = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_dir): print(f"Warning: Class directory not found: {class_dir}"); continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')): self.image_paths.append(os.path.join(class_dir, filename)); self.labels.append(self.class_labels_map[class_folder]); self.class_counts[class_folder] += 1
        self.image_size = image_size; self.split = split; self.base_transform_list = [transforms.Resize(image_size), transforms.Lambda(advanced_preprocess), transforms.ToTensor(), transforms.Lambda(to_float_tensor)]; self.normalize_transform = None
        self.augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20), transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.RandomGrayscale(p=0.1)])
    def set_normalization(self, normalize_transform): self.normalize_transform = normalize_transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L'); image = transforms.Compose(self.base_transform_list)(image_pil)
            if self.split == 'train' and self.apply_normalize: image = self.augment_transforms(image)
            if self.apply_normalize and self.normalize_transform: image = self.normalize_transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception: return torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float), torch.tensor(-1, dtype=torch.long)
class NotumorMRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train'):
        self.image_paths = []; notumor_dir = os.path.join(data_dir, 'notumor')
        if os.path.isdir(notumor_dir):
            for filename in os.listdir(notumor_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')): self.image_paths.append(os.path.join(notumor_dir, filename))
        self.image_size = image_size; self.split = split; self.transforms = transforms.Compose([transforms.Resize(image_size), transforms.Lambda(advanced_preprocess), transforms.ToTensor(), transforms.Lambda(to_float_tensor)]); self.augment_ae = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(degrees=5)])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            image = self.transforms(Image.open(self.image_paths[idx]).convert('L'))
            if self.split == 'train': image = self.augment_ae(image)
            return image, image
        except Exception: return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))

# --- Models ---
class EnhancedAutoencoder(nn.Module):
    def __init__(self, cbam_reduction_ratio=8, cbam_kernel_size=5):
        super(EnhancedAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_enc1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size); self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_enc2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size); self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_enc3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size); self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.cbam_bottleneck = CBAM(256, cbam_reduction_ratio, cbam_kernel_size)
        self.attention = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_dec3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_dec2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_dec1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x):
        e1_proc = self.enc1(x); e1 = self.cbam_enc1(e1_proc); p1 = self.pool1(e1)
        e2_proc = self.enc2(p1); e2 = self.cbam_enc2(e2_proc); p2 = self.pool2(e2)
        e3_proc = self.enc3(p2); e3 = self.cbam_enc3(e3_proc); p3 = self.pool3(e3)
        b_proc = self.bottleneck(p3); b = self.cbam_bottleneck(b_proc); att_map = self.attention(b)
        d3_up = self.up3(b); d3_cat = torch.cat([d3_up, e3], dim=1); d3_proc = self.dec3(d3_cat); d3 = self.cbam_dec3(d3_proc)
        d2_up = self.up2(d3); d2_cat = torch.cat([d2_up, e2], dim=1); d2_proc = self.dec2(d2_cat); d2 = self.cbam_dec2(d2_proc)
        d1_up = self.up1(d2); d1_cat = torch.cat([d1_up, e1], dim=1); d1_proc = self.dec1(d1_cat); d1 = self.cbam_dec1(d1_proc)
        reconstruction = torch.sigmoid(self.out_conv(d1)); return reconstruction, att_map

class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7):
        super(DenseNet121TumorClassificationModel, self).__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)
        original_conv0 = self.densenet121.features.conv0
        self.densenet121.features.conv0 = nn.Conv2d(1, original_conv0.out_channels, kernel_size=original_conv0.kernel_size, stride=original_conv0.stride, padding=original_conv0.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                self.densenet121.features.conv0.weight.copy_(original_conv0.weight.mean(dim=1, keepdim=True))
        num_ftrs = self.densenet121.classifier.in_features
        self.cbam = CBAM(num_ftrs, cbam_reduction_ratio, cbam_kernel_size)
        self.classifier_head = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.densenet121.classifier = nn.Identity()
        self._initialize_weights(self.classifier_head)
    def _initialize_weights(self, classifier_module):
        for m in classifier_module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.cbam(self.densenet121.features(x))
        out = F.adaptive_avg_pool2d(F.relu(features, inplace=True), (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier_head(out)

# --- Utility Functions ---
def calculate_mean_std(loader):
    mean, std, total_samples = 0., 0., 0
    for data in tqdm(loader, desc="Mean/Std Calc"):
        images = data[0]
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[1] == 1:
            batch_samples = images.size(0); images_flat = images.view(batch_samples, 1, -1); mean += images_flat.mean(2).sum(0); std += images_flat.std(2).sum(0); total_samples += batch_samples
    if total_samples == 0: return torch.tensor([0.5]), torch.tensor([0.5])
    mean /= total_samples; std /= total_samples; return mean, torch.max(std, torch.tensor(1e-6))

# --- Visualization Functions with Saving ---
def plot_training_history(history, save_path=None):
    if not history or not history['train_loss']: print("History is empty. Skipping plotting."); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss'); ax1.set_title('Model Loss vs. Epochs'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_f1'], label='Train F1-Score'); ax2.plot(history['val_f1'], label='Validation F1-Score'); ax2.set_title('Model F1-Score vs. Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1-Score (Weighted)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"Training history plot saved to {save_path}")
    plt.show(); plt.close(fig)
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred); fig = plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title('Confusion Matrix on Test Set'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    if save_path: plt.savefig(save_path); print(f"Confusion matrix saved to {save_path}")
    plt.show(); plt.close(fig)
def plot_misclassified_examples(y_true, y_pred, images_tensor, class_names, mean, std, max_examples=16, save_path=None):
    misclassified_idx = np.where(np.array(y_pred) != np.array(y_true))[0]
    if len(misclassified_idx) == 0: print("\nExcellent! No misclassifications found."); return
    print(f"\nFound {len(misclassified_idx)} misclassified examples. Showing/saving up to {max_examples}.")
    inv_normalize = transforms.Compose([transforms.Normalize(mean=[0.], std=[1/s for s in std]), transforms.Normalize(mean=[-m for m in mean], std=[1.])])
    fig = plt.figure(figsize=(15, 15)); num_to_show = min(len(misclassified_idx), max_examples)
    for i, idx in enumerate(misclassified_idx[:num_to_show]):
        plt.subplot(4, 4, i + 1); image = inv_normalize(images_tensor[idx]).cpu().numpy(); plt.imshow(np.transpose(image, (1, 2, 0)).squeeze(), cmap='gray'); plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", fontsize=10); plt.axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"Misclassified examples plot saved to {save_path}")
    plt.show(); plt.close(fig)

# --- Training Loops ---
def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path, attention_weight, ssim_weight):
    if os.path.exists(model_path): model.load_state_dict(torch.load(model_path, map_location=device)); return model
    model.to(device); criterion_mse, criterion_ssim = nn.MSELoss(), lambda p, t: 1.0 - ssim(p, t, data_range=1.0, size_average=True); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5); best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train(); total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            if torch.any(images.isnan()): continue
            images, targets = images.to(device), targets.to(device); optimizer.zero_grad(); reconstructed, attention_maps = model(images)
            loss = (1-ssim_weight)*criterion_mse(reconstructed, targets) + ssim_weight*criterion_ssim(reconstructed, targets) + attention_weight*torch.mean(torch.abs(attention_maps))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += loss.item()
        avg_loss = total_loss / len(train_loader); scheduler.step(avg_loss)
        print(f"AE Epoch {epoch+1} Loss: {avg_loss:.4f}")
        if avg_loss < best_loss: best_loss = avg_loss; torch.save(model.state_dict(), model_path); print(f"  New best AE saved to {model_path} (Loss: {best_loss:.4f})")
    model.load_state_dict(torch.load(model_path, map_location=device)); return model

def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_path):
    if os.path.exists(model_path):
        print(f"Loading existing classifier model from {model_path}. Skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device)); return model, {}, pd.DataFrame()
    model.to(device); best_val_f1 = 0.0; history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}; epoch_logs = []
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for epoch in range(num_epochs):
        model.train(); train_loss = 0.0; all_labels_train, all_preds_train = [], []
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            if torch.any(labels == -1): continue
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if isinstance(scheduler, OneCycleLR): scheduler.step()
            train_loss += loss.item(); _, predicted = torch.max(outputs.data, 1); all_labels_train.extend(labels.cpu().numpy()); all_preds_train.extend(predicted.cpu().numpy())
        avg_train_loss = train_loss/len(train_loader); train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', zero_division=0); history['train_loss'].append(avg_train_loss); history['train_f1'].append(train_f1)
        model.eval(); val_loss = 0.0; all_labels_val, all_preds_val = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                if torch.any(labels == -1): continue
                images, labels = images.to(device), labels.to(device); outputs = model(images); loss = criterion(outputs, labels); val_loss += loss.item(); _, predicted = torch.max(outputs.data, 1); all_labels_val.extend(labels.cpu().numpy()); all_preds_val.extend(predicted.cpu().numpy())
        avg_val_loss = val_loss/len(val_loader); val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0); history['val_loss'].append(avg_val_loss); history['val_f1'].append(val_f1)
        report_dict = classification_report(all_labels_val, all_preds_val, target_names=class_names, output_dict=True, zero_division=0)
        log_entry = {'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_f1_weighted': train_f1, 'val_f1_weighted': val_f1, 'val_accuracy': report_dict['accuracy']}
        for class_name in class_names:
            log_entry[f'val_precision_{class_name}'] = report_dict[class_name]['precision']; log_entry[f'val_recall_{class_name}'] = report_dict[class_name]['recall']; log_entry[f'val_f1_{class_name}'] = report_dict[class_name]['f1-score']
        epoch_logs.append(log_entry)
        print(f"\n{classification_report(all_labels_val, all_preds_val, target_names=class_names, zero_division=0)}")
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f} F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; torch.save(model.state_dict(), model_path); print(f"  New best classifier saved to {model_path} (F1: {best_val_f1:.4f})")
    log_df = pd.DataFrame(epoch_logs)
    model.load_state_dict(torch.load(model_path, map_location=device)); return model, history, log_df

# --- Final Test Set Evaluation with Saving ---
def evaluate_classifier_on_test_set(model, test_loader, device, class_names, train_mean, train_std, results_dir):
    print("\n--- Evaluating Final Model on Test Set ---")
    model.to(device); model.eval(); all_labels, all_preds, all_images = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Set Evaluation"):
            if torch.any(labels == -1): continue
            all_images.append(images.cpu()); all_labels.extend(labels.cpu().numpy())
            outputs = model(images.to(device)); _, predicted = torch.max(outputs.data, 1); all_preds.extend(predicted.cpu().numpy())
    if not all_labels: print("Test set empty. Cannot evaluate."); return
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_path = os.path.join(results_dir, "test_set_report.txt")
    with open(report_path, 'w') as f: f.write("--- Test Set Performance Report ---\n\n" + report_str)
    print(f"Test report saved to {report_path}\n{report_str}")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=os.path.join(results_dir, "confusion_matrix.png"))
    all_images_tensor = torch.cat(all_images, dim=0)
    plot_misclassified_examples(all_labels, all_preds, all_images_tensor, class_names, train_mean.cpu().numpy(), train_std.cpu().numpy(), save_path=os.path.join(results_dir, "misclassified_examples.png"))

# --- Tumor Outlining Functions ---
def enhance_outline_detection(rec_err, att_map, threshold=None, min_solidity=0.6):
    rec_norm=(rec_err-rec_err.min())/(rec_err.max()-rec_err.min()+1e-8); att_norm=(att_map-att_map.min())/(att_map.max()-att_map.min()+1e-8)
    combined=cv2.medianBlur(cv2.GaussianBlur((np.clip(0.4*rec_norm+0.6*att_norm,0,1)*255).astype(np.uint8),(5,5),0),5)
    if threshold is None: threshold, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) if combined.max() > 0 else (255, 0)
    mask=cv2.morphologyEx(cv2.morphologyEx((combined > threshold).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))),cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); valid_contours=[]
    for c in contours:
        area=cv2.contourArea(c); hull_area=cv2.contourArea(cv2.convexHull(c))
        if 50 < area < (mask.shape[0]*mask.shape[1]*0.6) and hull_area > 0 and (area/hull_area) >= min_solidity: valid_contours.append(c)
    return mask, (combined.astype(np.float32)/255.0), valid_contours
def outline_tumor_enhanced(orig_pil, in_tensor, recon_tensor, att_tensor, size, err_thresh, solidity, show_plots):
    in_np, recon_np, att_np = in_tensor.squeeze().cpu().numpy(), recon_tensor.squeeze().cpu().numpy(), att_tensor.squeeze().cpu().numpy()
    orig_resized_np = np.array(orig_pil.resize(size)); rec_err = np.abs(in_np - recon_np)
    att_resized = F.interpolate(torch.from_numpy(att_np).unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    mask, combined_map, contours = enhance_outline_detection(rec_err, att_resized, err_thresh, solidity)
    outlined_img = cv2.cvtColor(orig_resized_np, cv2.COLOR_GRAY2BGR)
    if contours:
        overlay = outlined_img.copy(); cv2.drawContours(overlay, contours, -1, (0,255,0), cv2.FILLED); cv2.addWeighted(overlay, 0.3, outlined_img, 0.7, 0, outlined_img); cv2.drawContours(outlined_img, contours, -1, (255,0,0), 2)
    if show_plots:
        fig, axes = plt.subplots(1, 8, figsize=(20, 5)); titles = ['Original', 'AE Input', 'AE Recon', 'Error', 'Attention', 'Combined', 'Mask', 'Result']
        imgs = [orig_resized_np, in_np, recon_np, rec_err, att_resized, combined_map, mask, cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB)]; cmaps = ['gray', 'gray', 'gray', 'hot', 'viridis', 'hot', 'gray', None]
        for ax, img, title, cmap in zip(axes, imgs, titles, cmaps): ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis('off')
        plt.tight_layout(); plt.show(); plt.close(fig)
    return Image.fromarray(orig_resized_np), Image.fromarray(cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB))

# --- Prediction Function with Saving ---
def predict_and_outline_simple(input_image_path, class_model_path, ae_model_path, image_size, norm_transform,
                            device_classifier, device_outlining, save_final_image_path=None, **kwargs):
    if not os.path.exists(input_image_path): print(f"Error: Not found {input_image_path}"); return None, None, None
    orig_pil = Image.open(input_image_path).convert('L')
    classify_tensor = transforms.Compose([transforms.Resize(image_size), transforms.Lambda(advanced_preprocess), transforms.ToTensor(), transforms.Lambda(to_float_tensor), norm_transform])(orig_pil).unsqueeze(0).to(device_classifier)
    classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False); classifier.load_state_dict(torch.load(class_model_path, map_location=device_classifier)); classifier.to(device_classifier).eval(); class_map = {0:'glioma', 1:'meningioma', 2:'notumor', 3:'pituitary'}
    with torch.no_grad(): pred = classifier(classify_tensor); probs = torch.softmax(pred, 1); conf, idx = torch.max(probs, 1); pred_name = class_map[idx.item()]; conf = conf.item()
    print(f"Prediction: {pred_name} (Confidence: {conf:.2%})")
    outlined_pil = orig_pil.resize(image_size)
    if pred_name != 'notumor' and os.path.exists(ae_model_path):
        ae_tensor = transforms.Compose([transforms.Resize(image_size), transforms.Lambda(advanced_preprocess), transforms.ToTensor(), transforms.Lambda(to_float_tensor)])(orig_pil).unsqueeze(0).to(device_outlining)
        autoencoder = EnhancedAutoencoder(); autoencoder.load_state_dict(torch.load(ae_model_path, map_location=device_outlining)); autoencoder.to(device_outlining).eval()
        with torch.no_grad(): recon, att = autoencoder(ae_tensor)
        
        # Extract arguments for outline_tumor_enhanced
        err_thresh = kwargs.get('error_threshold', None)
        solidity = kwargs.get('min_solidity', 0.6)
        show_plots_outline = kwargs.get('show_plots', False)

        _, outlined_pil = outline_tumor_enhanced(orig_pil, ae_tensor.squeeze(0), recon.squeeze(0), att.squeeze(0), image_size, err_thresh, solidity, show_plots_outline)
    if save_final_image_path and outlined_pil: outlined_pil.save(save_final_image_path); print(f"Final prediction image saved to {save_final_image_path}")
    if kwargs.get('show_final_plot', True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)); ax1.imshow(orig_pil.resize(image_size), cmap='gray'); ax1.set_title("Original"); ax1.axis('off'); ax2.imshow(outlined_pil); ax2.set_title(f"Result: {pred_name.capitalize()} ({conf:.2%})"); ax2.axis('off'); plt.tight_layout(); plt.show(); plt.close(fig)
    return pred_name, conf, outlined_pil

# --- Main Execution ---
def main():
    freeze_support()
    device = get_device()
    data_dir = 'E:\\project\\dataset1\\training1' # <<< CHECK/CHANGE
    results_dir = 'classification_results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Configs
    image_size, batch_size, num_epochs_c, num_epochs_ae, num_workers = (256,256), 16, 25, 25, 2
    c_path = os.path.join(results_dir, 'densenet121_cbam_best.pth')
    ae_path = os.path.join(results_dir, 'enhanced_ae_notumor_best.pth')

    # Data Loading & Split
    full_ds_for_stats = MRIDataset(data_dir, image_size, 'full', False)
    if len(full_ds_for_stats) == 0: print("Error: No images found."); return
    labels = full_ds_for_stats.labels
    train_idx, temp_idx = train_test_split(range(len(full_ds_for_stats)), test_size=0.3, random_state=42, stratify=labels if min(Counter(labels).values()) > 1 else None)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_idx] if min(Counter([labels[i] for i in temp_idx]).values()) > 1 else None)

    # Normalization & DataLoaders
    mean, std = calculate_mean_std(DataLoader(torch.utils.data.Subset(full_ds_for_stats, train_idx), batch_size, num_workers=num_workers))
    norm = transforms.Normalize(mean=mean, std=std)
    class_ds_final = MRIDataset(data_dir, image_size, 'full', True); class_ds_final.set_normalization(norm)
    train_c_loader = DataLoader(torch.utils.data.Subset(class_ds_final, train_idx), batch_size, True, num_workers=num_workers, drop_last=True)
    val_c_loader = DataLoader(torch.utils.data.Subset(class_ds_final, val_idx), batch_size, False, num_workers=num_workers)
    test_c_loader = DataLoader(torch.utils.data.Subset(class_ds_final, test_idx), batch_size, False, num_workers=num_workers)

    # Autoencoder Training
    ae_ds = NotumorMRIDataset(data_dir, image_size);
    if len(ae_ds) > 0:
        print("\n--- Starting Autoencoder Training ---")
        train_ae_loader = DataLoader(ae_ds, batch_size, True, num_workers=num_workers, drop_last=True);
        train_enhanced_autoencoder(EnhancedAutoencoder(), train_ae_loader, num_epochs_ae, device, ae_path, 0.03, 0.20)

    # Classifier Training
    print("\n--- Starting Classifier Training ---")
    classifier = DenseNet121TumorClassificationModel(num_classes=4); train_labels = [labels[i] for i in train_idx]; class_counts = Counter(train_labels)
    weights = torch.tensor([len(train_labels) / class_counts[i] for i in range(4)], dtype=torch.float).to(device) if len(class_counts) == 4 and all(v>0 for v in class_counts.values()) else torch.ones(4, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights); optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5); scheduler = OneCycleLR(optimizer, 5e-4, total_steps=num_epochs_c*len(train_c_loader))
    trained_classifier, history, log_df = train_classifier(classifier, train_c_loader, val_c_loader, criterion, optimizer, scheduler, num_epochs_c, device, c_path)

    # Save Logs and Visualizations
    if not log_df.empty:
        log_df.to_excel(os.path.join(results_dir, "training_log_per_epoch.xlsx"), index=False)
        print(f"\nEpoch-by-epoch training log saved to {os.path.join(results_dir, 'training_log_per_epoch.xlsx')}")
    plot_training_history(history, save_path=os.path.join(results_dir, "training_history.png"))
    evaluate_classifier_on_test_set(trained_classifier, test_c_loader, device, ['glioma','meningioma','notumor','pituitary'], mean, std, results_dir)
    
    # Prediction on Sample Image
    sample_image_path = "C:\\Users\\Harish\\Downloads\\PituitaryAdenoma.jpeg" # <<< CHECK/CHANGE
    if os.path.exists(sample_image_path):
        print("\n--- Performing Prediction on Sample Image ---")
        predict_and_outline_simple(sample_image_path, c_path, ae_path, image_size, norm, device, torch.device('cpu'),
                                   save_final_image_path=os.path.join(results_dir, "sample_prediction_outlined.png"),
                                   error_threshold=None, min_solidity=0.65, show_plots=False, show_final_plot=True)
    else: print(f"\nSample image not found: {sample_image_path}. Skipping final prediction.")
    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()