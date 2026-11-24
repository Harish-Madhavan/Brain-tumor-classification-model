import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pytorch_msssim import ssim


def _enable_speed_flags(device=None):
    # NOTE: built-in torch.xpu backend only (no IPEX - deprecated).
    # cudnn.benchmark is CUDA-only; oneDNN autotuning on XPU is automatic.
    try:
        dev_type = getattr(device, 'type', None)
        if dev_type == 'cuda':
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _amp_dtype_for(device, override=None):
    """Best autocast dtype per backend: bf16 on XPU/CPU (stable, no scaler),
    fp16 on CUDA (needs GradScaler). Override with 'bf16'/'fp16'."""
    if override in ('bf16', 'bf16:'):
        return torch.bfloat16
    if override == 'fp16':
        return torch.float16
    try:
        dev_type = getattr(device, 'type', 'cpu')
    except Exception:
        dev_type = 'cpu'
    if dev_type == 'cuda':
        return torch.float16
    # XPU (Arc/iGPU) + CPU: bfloat16 is faster and numerically safer than fp16
    return torch.bfloat16


def _autocast_ctx(device, enabled=True, amp_dtype=None):
    """Device-aware autocast using the built-in backend (torch.amp, no IPEX)."""
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    try:
        dtype = _amp_dtype_for(device, override=amp_dtype)
        return torch.amp.autocast(device.type, dtype=dtype)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def _make_scaler(device, enabled=True, amp_dtype=None):
    # GradScaler is only needed for fp16 (CUDA default). bf16 (XPU/CPU
    # default) does not need scaling - returning None avoids overhead/bugs.
    if not enabled:
        return None
    try:
        dev_type = getattr(device, 'type', 'cpu')
    except Exception:
        return None
    if dev_type != 'cuda':
        return None
    if _amp_dtype_for(device, override=amp_dtype) != torch.float16:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.amp.GradScaler('cuda')
    except Exception:
        return None


def _empty_accel_cache(device):
    # iGPUs share system RAM - clearing cached blocks avoids pressure buildup.
    try:
        if getattr(device, 'type', None) == 'xpu' and hasattr(torch, 'xpu'):
            torch.xpu.empty_cache()
        elif getattr(device, 'type', None) == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        pass


def _maybe_channels_last(model, device, enabled=True):
    """Intel XPU (and CUDA) convs are faster in channels_last. Safe no-op fallback."""
    try:
        if enabled and getattr(device, 'type', None) in ('xpu', 'cuda'):
            model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model


def _to_device_nchw(batch, device, channels_last=True, non_blocking=True):
    t = batch.to(device, non_blocking=non_blocking)
    if channels_last and getattr(device, 'type', None) in ('xpu', 'cuda'):
        try:
            t = t.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return t


def compute_class_weights(labels, num_classes=4, device='cpu'):
    """Inverse-frequency weights to counter notumor over-representation."""
    counts = Counter([l for l in labels if l is not None and l >= 0])
    total = sum(counts.values()) or 1
    weights = [(total / (num_classes * counts.get(c, 1))) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _mixup_batch(images, labels, alpha=0.2):
    if alpha <= 0:
        return images, labels, None, None, None
    import torch.distributions.beta as _beta
    lam = float(_beta.Beta(alpha, alpha).sample().item())
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam, idx

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

def train_enhanced_autoencoder(model, train_loader, num_epochs, device, model_path, attention_weight, ssim_weight,
                               use_amp=True, amp_dtype=None, channels_last=True,
                               val_loader=None, early_stop_patience=6):
    eff_dtype = _amp_dtype_for(device, override=amp_dtype) if use_amp else None
    print(f"\n--- Training Enhanced Autoencoder for {num_epochs} Epochs (AMP={use_amp} dtype={eff_dtype} backend={device.type}) ---")
    _enable_speed_flags(device)
    model = _maybe_channels_last(model, device, enabled=channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = _make_scaler(device, enabled=use_amp, amp_dtype=amp_dtype)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train(); train_loss = 0.0; valid_samples = 0
        for data, targets in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{num_epochs}"):
            # Skip corrupt samples surfaced with -1.0 sentinel by NotumorMRIDataset
            valid = targets[:, 0, 0, 0] >= 0
            if not bool(valid.any()):
                continue
            inputs = _to_device_nchw(data[valid], device, channels_last=channels_last)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                reconstructed, att_map = model(inputs)

                l1_loss = nn.L1Loss()(reconstructed, inputs)
                # pytorch-msssim SSIM only supports fp32 -> cast explicitly so
                # XPU bf16 / CUDA fp16 autocast does not break or lose accuracy.
                ssim_loss = 1.0 - ssim(reconstructed.float(), inputs.float(), data_range=1.0, size_average=True)
                att_loss = torch.mean(att_map ** 2)

                loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss + attention_weight * att_loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            valid_samples += inputs.size(0)

        epoch_loss = train_loss / max(1, valid_samples)

        monitored_loss = epoch_loss
        if val_loader is not None:
            model.eval(); val_loss = 0.0; val_samples = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    valid = targets[:, 0, 0, 0] >= 0
                    if not bool(valid.any()):
                        continue
                    inputs = _to_device_nchw(data[valid], device, channels_last=channels_last)
                    with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                        reconstructed, att_map = model(inputs)
                        l1_loss = nn.L1Loss()(reconstructed, inputs)
                        ssim_loss = 1.0 - ssim(reconstructed.float(), inputs.float(), data_range=1.0, size_average=True)
                        att_loss = torch.mean(att_map ** 2)
                        loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss + attention_weight * att_loss
                    val_loss += loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)
            if val_samples > 0:
                monitored_loss = val_loss / val_samples
                print(f"Epoch [{epoch+1}/{num_epochs}] AE Train Loss: {epoch_loss:.6f} | Val Loss: {monitored_loss:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] AE Train Loss: {epoch_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] AE Loss: {epoch_loss:.6f}")

        scheduler.step(monitored_loss)
        _empty_accel_cache(device)

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best autoencoder weights to {model_path}")
        elif val_loader is not None:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"AE early stopping at epoch {epoch+1} (best Val Loss: {best_loss:.6f})")
                break

def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_path,
                     use_amp=True, grad_clip=1.0, early_stop_patience=6, mixup_alpha=0.0, amp_dtype=None, channels_last=True):
    eff_dtype = _amp_dtype_for(device, override=amp_dtype) if use_amp else None
    print(f"\n--- Training Classifier for {num_epochs} Epochs (AMP={use_amp} dtype={eff_dtype} backend={device.type}, mixup={mixup_alpha}) ---")
    _enable_speed_flags(device)
    model = _maybe_channels_last(model, device, enabled=channels_last)
    best_acc = 0.0; epochs_no_improve = 0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    scaler = _make_scaler(device, enabled=use_amp, amp_dtype=amp_dtype)

    for epoch in range(num_epochs):
        model.train(); running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            # Skip corrupt samples surfaced as label -1 by MRIDataset
            valid = labels >= 0
            if not bool(valid.any()):
                continue
            images = _to_device_nchw(images[valid], device, channels_last=channels_last)
            labels = labels[valid].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0 and images.size(0) > 1:
                images, y_a, y_b, lam, _ = _mixup_batch(images, labels, alpha=mixup_alpha)
                with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
            else:
                with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.detach(), 1)
            # For mixup, accuracy vs y_a is approximate; fine for logging
            total += labels.size(0); correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        history['train_loss'].append(epoch_loss); history['train_acc'].append(epoch_acc)

        # Validation (autocast on for XPU/CUDA speed; fp32 loss math unaffected)
        model.eval(); val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                valid = labels >= 0
                if not bool(valid.any()):
                    continue
                images = _to_device_nchw(images[valid], device, channels_last=channels_last)
                labels = labels[valid].to(device, non_blocking=True)
                with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0); val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / max(1, val_total)
        val_epoch_acc = val_correct / max(1, val_total)
        history['val_loss'].append(val_epoch_loss); history['val_acc'].append(val_epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_epoch_loss)
        else:
            scheduler.step()
        _empty_accel_cache(device)

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best classifier weights to {model_path} (Val Acc: {val_epoch_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (best Val Acc: {best_acc:.4f})")
                break

    return history

def evaluate_classifier_on_test_set(model, test_loader, device, class_names, train_mean, train_std, results_dir,
                                    use_amp=True, amp_dtype=None, channels_last=True):
    print("\n--- Evaluating Classifier on Test Set ---")
    model = _maybe_channels_last(model, device, enabled=channels_last)
    model.eval(); all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            valid = labels >= 0
            if not bool(valid.any()):
                continue
            images, labels = images[valid], labels[valid]
            all_images.append(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            inputs = _to_device_nchw(images, device, channels_last=channels_last)
            with _autocast_ctx(device, enabled=use_amp, amp_dtype=amp_dtype):
                outputs = model(inputs)
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
