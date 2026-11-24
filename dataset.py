import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Reuse a single CLAHE object per process (cv2.createCLAHE is expensive).
_CLAHE = None

def _get_clahe():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE

def advanced_preprocess(img):
    img_np = np.array(img)
    img_clahe = _get_clahe().apply(img_np)
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return Image.fromarray(img_blur)

def to_float_tensor(img):
    return img.float()


def _quiet_blas_worker(worker_id):
    # Each DataLoader worker inherits multi-threaded BLAS (OpenBLAS/MKL).
    # Cap to 1 thread per worker to avoid oversubscription -> OpenBLAS
    # "Memory allocation failed" and swap death on shared-RAM iGPUs.
    try:
        import os as _os
        _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        _os.environ.setdefault('OMP_NUM_THREADS', '1')
        import torch as _t
        _t.set_num_threads(1)
        _t.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass


def default_loader_kwargs(batch_size=32, num_workers=None, device=None):
    """Speed-tuned DataLoader kwargs with XPU/iGPU-safe fallbacks.

    Intel iGPUs share system RAM with the CPU: 8 spawn workers x prefetch x
    pinned memory easily OOMs host RAM -> OpenBLAS allocation failure. So on
    XPU (or Windows spawn) default to 0 workers unless the user overrides
    with --workers. pin_memory only helps discrete CUDA; on unified-memory
    XPU it just wastes RAM, so it is CUDA-only now.
    """
    import os as _os
    import sys as _sys
    dev_type = getattr(device, 'type', None) if device is not None else None
    if num_workers is None:
        if dev_type == 'xpu':
            num_workers = 0
        elif _sys.platform.startswith('win'):
            num_workers = 0
        else:
            try:
                cpu_n = _os.cpu_count() or 4
            except Exception:
                cpu_n = 4
            num_workers = min(4, max(0, cpu_n - 2))
    kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    if num_workers and num_workers > 0:
        # persistent_workers + prefetch keep workers alive (fast) but hold
        # RAM; only use them when the user explicitly asked for workers.
        kwargs.update(persistent_workers=True, prefetch_factor=2,
                      worker_init_fn=_quiet_blas_worker)
    try:
        import torch as _torch
        # CUDA-only: XPU iGPU is unified memory, pinned host buffers are pure overhead.
        if dev_type == 'cuda' or (dev_type is None and _torch.cuda.is_available()):
            kwargs.update(pin_memory=True)
    except Exception:
        pass
    return kwargs

class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', apply_normalize=True, indices=None):
        self.image_paths, self.labels = [], []
        class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_labels_map = {folder: i for i, folder in enumerate(class_folders)}
        self.class_counts = {folder: 0 for folder in class_folders}
        self.apply_normalize = apply_normalize
        
        for class_folder in class_folders:
            class_dir = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_labels_map[class_folder])

        # Optional deterministic subset (used for stratified train/val/test splits).
        # This fixes the previous bug where train/val/test all loaded the full dataset.
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        for lbl in self.labels:
            self.class_counts[class_folders[lbl]] += 1
                    
        self.image_size = image_size
        self.split = split
        self.base_transform_list = [
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor)
        ]
        # Pre-compose once (was rebuilt on every __getitem__ -> wasted CPU).
        self.base_compose = transforms.Compose(self.base_transform_list)
        self.normalize_transform = None
        # Lighter, MRI-appropriate augmentation: Affine/Rotation dominate CPU cost,
        # so shrink ranges and drop no-op RandomGrayscale on 1-channel data.
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
        
    def set_normalization(self, normalize_transform):
        self.normalize_transform = normalize_transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = self.base_compose(image_pil)
            if self.split == 'train' and self.apply_normalize:
                image = self.augment_transforms(image)
            if self.apply_normalize and self.normalize_transform:
                image = self.normalize_transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float), torch.tensor(-1, dtype=torch.long)

class NotumorMRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', indices=None):
        self.image_paths = []
        notumor_dir = os.path.join(data_dir, 'notumor')
        if os.path.isdir(notumor_dir):
            for filename in sorted(os.listdir(notumor_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(notumor_dir, filename))
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
        self.image_size = image_size
        self.split = split
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor)
        ])
        self.augment_ae = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5)
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            image = self.transforms(Image.open(self.image_paths[idx]).convert('L'))
            if self.split == 'train':
                image = self.augment_ae(image)
            return image, image
        except Exception:
            zeros = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float)
            sentinel = torch.full((1, self.image_size[0], self.image_size[1]), -1.0, dtype=torch.float)
            return zeros, sentinel


def seed_everything(seed=42):
    """Seed stdlib, numpy and torch (incl. cuda/xpu when present) for
    reproducible splits, shuffling, augmentation and MixUp."""
    import random
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except Exception:
        pass


def stratified_train_val_test_split(labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Deterministic stratified split. Returns (train_idx, val_idx, test_idx).

    Falls back to non-stratified split when a class has <2 samples.
    """
    from collections import Counter
    from sklearn.model_selection import train_test_split

    n = len(labels)
    all_idx = list(range(n))
    if n == 0:
        return [], [], []
    try:
        stratify = labels if min(Counter(labels).values()) > 1 else None
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=(val_ratio + test_ratio), random_state=seed, stratify=stratify)
        if not temp_idx:
            return train_idx, [], []
        # Relative test fraction within temp pool
        rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
        temp_labels = [labels[i] for i in temp_idx]
        strat_temp = temp_labels if (temp_labels and min(Counter(temp_labels).values()) > 1) else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=rel_test, random_state=seed, stratify=strat_temp)
        return list(train_idx), list(val_idx), list(test_idx)
    except ValueError:
        # Extremely small datasets: simple sequential split
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        return all_idx[:n - n_val - n_test], all_idx[n - n_val - n_test:n - n_test], all_idx[n - n_test:]
