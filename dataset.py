import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps

def equalize_histogram(img):
    return ImageOps.equalize(img)

def advanced_preprocess(img):
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return Image.fromarray(img_blur)

def to_float_tensor(img):
    return img.float()

class MRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train', apply_normalize=True):
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
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_labels_map[class_folder])
                    self.class_counts[class_folder] += 1
                    
        self.image_size = image_size
        self.split = split
        self.base_transform_list = [
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor)
        ]
        self.normalize_transform = None
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomGrayscale(p=0.1)
        ])
        
    def set_normalization(self, normalize_transform):
        self.normalize_transform = normalize_transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        try:
            image_pil = Image.open(image_path).convert('L')
            image = transforms.Compose(self.base_transform_list)(image_pil)
            if self.split == 'train' and self.apply_normalize:
                image = self.augment_transforms(image)
            if self.apply_normalize and self.normalize_transform:
                image = self.normalize_transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float), torch.tensor(-1, dtype=torch.long)

class NotumorMRIDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), split='train'):
        self.image_paths = []
        notumor_dir = os.path.join(data_dir, 'notumor')
        if os.path.isdir(notumor_dir):
            for filename in os.listdir(notumor_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(notumor_dir, filename))
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
            return torch.zeros((1, self.image_size[0], self.image_size[1])), torch.zeros((1, self.image_size[0], self.image_size[1]))
