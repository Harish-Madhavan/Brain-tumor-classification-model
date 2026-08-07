import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pytorch_msssim import ssim

from dataset import MRIDataset
from models import CBAM, DenseNet121TumorClassificationModel, EnhancedAutoencoder

def calculate_mean_std(loader):
    print("Calculating training dataset Mean and Std for normalization...")
    sum_val, squared_sum, total_batches = 0.0, 0.0, 0
    for data, _ in tqdm(loader, desc="Mean/Std Calc"):
        sum_val += torch.mean(data, dim=[0, 2, 3])
        squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        total_batches += 1
    mean = sum_val / total_batches
    std = (squared_sum / total_batches - mean ** 2) ** 0.5
    return mean, std

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Train')
    if 'val_loss' in history: ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss History'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train')
        if 'val_acc' in history: ax2.plot(history['val_acc'], label='Val')
        ax2.set_title('Accuracy History'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"History plot saved to {save_path}")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label'); ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)

def plot_misclassified_examples(y_true, y_pred, images_tensor, class_names, mean, std, max_examples=16, save_path=None):
    mis_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if not mis_idx: print("No misclassified examples found."); return
    n_show = min(len(mis_idx), max_examples)
    cols = 4; rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten() if n_show > 1 else [axes]
    for idx, ax in zip(mis_idx[:n_show], axes):
        img = images_tensor[idx].squeeze().numpy() * std + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray'); ax.set_title(f"T: {class_names[y_true[idx]]}\nP: {class_names[y_pred[idx]]}", fontsize=9); ax.axis('off')
    for ax in axes[n_show:]: ax.axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"Misclassified examples saved to {save_path}")
    plt.close(fig)

def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path, attention_weight, ssim_weight):
    print(f"\n--- Training Enhanced Autoencoder for {num_epochs} Epochs ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train(); train_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{num_epochs}"):
            inputs = data.to(device)
            optimizer.zero_grad()
            reconstructed, att_map = model(inputs)
            
            l1_loss = nn.L1Loss()(reconstructed, inputs)
            ssim_loss = 1.0 - ssim(reconstructed, inputs, data_range=1.0, size_average=True)
            att_loss = torch.mean(att_map ** 2)
            
            loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss + attention_weight * att_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        epoch_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] AE Loss: {epoch_loss:.6f}")
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best autoencoder weights to {model_path}")

def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_path):
    print(f"\n--- Training Classifier for {num_epochs} Epochs ---")
    best_acc = 0.0; history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    
    for epoch in range(num_epochs):
        model.train(); running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0); correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss); history['train_acc'].append(epoch_acc)
        
        # Validation
        model.eval(); val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0); val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss); history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_epoch_loss)
        else:
            scheduler.step()
            
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best classifier weights to {model_path} (Val Acc: {val_epoch_acc:.4f})")
            
    return history

def evaluate_classifier_on_test_set(model, test_loader, device, class_names, train_mean, train_std, results_dir):
    print("\n--- Evaluating Classifier on Test Set ---")
    model.eval(); all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            all_images.append(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    if not all_labels:
        print("Test set empty. Cannot evaluate.")
        return
        
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_path = os.path.join(results_dir, "test_set_report.txt")
    with open(report_path, 'w') as f:
        f.write("--- Test Set Performance Report ---\n\n" + report_str)
        
    print(f"Test report saved to {report_path}\n{report_str}")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=os.path.join(results_dir, "confusion_matrix.png"))
    all_images_tensor = torch.cat(all_images, dim=0)
    plot_misclassified_examples(
        all_labels, all_preds, all_images_tensor, class_names,
        train_mean.cpu().numpy(), train_std.cpu().numpy(),
        save_path=os.path.join(results_dir, "misclassified_examples.png")
    )
