import json
import os
import traceback
from contextlib import nullcontext

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset import advanced_preprocess, to_float_tensor
from explainability import GradCAM, get_gradcam_overlay
from localization import outline_tumor_enhanced
from models import DenseNet121TumorClassificationModel, EnhancedAutoencoder

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}
DEFAULT_MEAN = [0.1834]
DEFAULT_STD = [0.1982]
METADATA_FILENAME = 'model_metadata.json'


def get_device():
    """Select preferred compute device: Intel XPU -> CUDA -> CPU."""
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            n = torch.xpu.device_count() if hasattr(torch.xpu, 'device_count') else 1
            print(f"Using XPU device (built-in backend, no IPEX). Count: {n}")
            return torch.device("xpu")
    except Exception as e:
        print(f"XPU check failed ({e}), falling back.")
    if torch.cuda.is_available():
        print("XPU not available, using CUDA device.")
        return torch.device("cuda")
    print("XPU and CUDA not available, using CPU device.")
    return torch.device("cpu")


def infer_channels_last(device):
    return getattr(device, 'type', None) in ('xpu', 'cuda')


def _get_autocast_ctx(device):
    if not infer_channels_last(device):
        return nullcontext()
    try:
        from train import _autocast_ctx
        return _autocast_ctx(device, enabled=True)
    except Exception:
        return nullcontext()


def safe_torch_load(path, map_location='cpu'):
    """Load a PyTorch state_dict using weights_only=True when supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_path(custom_path, default_filename):
    if custom_path and os.path.exists(custom_path):
        return custom_path

    res_path = os.path.join('classification_results', default_filename)
    if os.path.exists(res_path):
        return res_path

    if os.path.exists(default_filename):
        return default_filename

    if 'ae' in default_filename.lower() or 'autoencoder' in default_filename.lower():
        alternatives = [
            'enhanced_ae_notumor_best.pth',
            'enhanced_autoencoder_notumor_cbam_ssim_v2_best.pth',
            'enhanced_autoencoder_notumor_cbam_best.pth',
            'enhanced_autoencoder_notumor.pth',
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                return alt
            alt_res = os.path.join('classification_results', alt)
            if os.path.exists(alt_res):
                return alt_res

    if 'densenet' in default_filename.lower() or 'cbam' in default_filename.lower():
        alternatives = ['densenet121_cbam_best.pth']
        for alt in alternatives:
            if os.path.exists(alt):
                return alt
            alt_res = os.path.join('classification_results', alt)
            if os.path.exists(alt_res):
                return alt_res

    return res_path


def save_model_metadata(results_dir, mean, std, image_size=(256, 256), class_names=None):
    """Persist normalization stats and image configuration alongside weights."""
    os.makedirs(results_dir, exist_ok=True)
    mean_list = [float(x) for x in torch.as_tensor(mean).flatten().tolist()]
    std_list = [float(x) for x in torch.as_tensor(std).flatten().tolist()]
    payload = {
        'mean': mean_list,
        'std': std_list,
        'image_size': list(image_size),
        'class_names': list(class_names or CLASS_NAMES),
    }
    meta_path = os.path.join(results_dir, METADATA_FILENAME)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return meta_path


def load_model_metadata(results_dir):
    """Load persisted normalization stats if present, else return None."""
    meta_path = os.path.join(results_dir, METADATA_FILENAME)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {meta_path}: {e}")
        return None


def load_classifier(c_path, device_classifier):
    clf = DenseNet121TumorClassificationModel(num_classes=len(CLASS_NAMES), pretrained=False)
    clf.load_state_dict(safe_torch_load(c_path, map_location=device_classifier))
    clf.to(device_classifier)
    if infer_channels_last(device_classifier):
        try:
            clf = clf.to(memory_format=torch.channels_last)
        except Exception:
            pass
    clf.eval()
    return clf


def load_autoencoder(ae_path, device_outlining):
    if not ae_path or not os.path.exists(ae_path):
        return None
    try:
        ae = EnhancedAutoencoder()
        ae.load_state_dict(safe_torch_load(ae_path, map_location=device_outlining))
        ae.to(device_outlining)
        if infer_channels_last(device_outlining):
            try:
                ae = ae.to(memory_format=torch.channels_last)
            except Exception:
                pass
        ae.eval()
        return ae
    except Exception as e:
        print(f"Warning: Failed to load autoencoder weights: {e}")
        return None


class DiagnosticPipeline:
    """Shared inference pipeline for CLI, batch, and Gradio web UI."""

    def __init__(self, classifier, autoencoder, image_size, norm_transform,
                 device_classifier, device_outlining):
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.image_size = tuple(image_size)
        self.norm_transform = norm_transform
        self.device_classifier = device_classifier
        self.device_outlining = device_outlining

        base_steps = [
            transforms.Resize(self.image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
        ]
        self.ae_transform = transforms.Compose(base_steps)
        self.classify_transform = transforms.Compose(base_steps + [norm_transform])

    @classmethod
    def from_paths(cls, class_model_path, ae_model_path, image_size, norm_transform,
                   device_classifier, device_outlining):
        classifier = load_classifier(class_model_path, device_classifier)
        autoencoder = load_autoencoder(ae_model_path, device_outlining)
        return cls(classifier, autoencoder, image_size, norm_transform,
                   device_classifier, device_outlining)

    def _prepare_tensor(self, transform, orig_pil, device):
        tensor = transform(orig_pil).unsqueeze(0).to(device)
        if infer_channels_last(device):
            try:
                tensor = tensor.to(memory_format=torch.channels_last)
            except Exception:
                pass
        return tensor

    def predict(self, orig_pil, generate_gradcam=True, err_thresh=None,
                min_solidity=0.6, pixel_spacing_mm=1.15, show_plots=False):
        if orig_pil.mode != 'L':
            orig_pil = orig_pil.convert('L')

        classify_tensor = self._prepare_tensor(
            self.classify_transform, orig_pil, self.device_classifier
        )

        self.classifier.eval()
        with torch.no_grad(), _get_autocast_ctx(self.device_classifier):
            pred = self.classifier(classify_tensor)
            probs = torch.softmax(pred.float(), 1).squeeze(0)
            conf, idx = torch.max(probs, 0)
            pred_idx = int(idx.item())
            pred_name = CLASS_MAP[pred_idx]
            conf_val = float(conf.item())

        prob_dict = {CLASS_NAMES[i].capitalize(): float(probs[i].item()) for i in range(len(CLASS_NAMES))}

        gradcam_pil = None
        if generate_gradcam:
            with torch.enable_grad():
                gradcam_tensor = classify_tensor.clone().detach()
                target_layer = self.classifier.densenet121.features.norm5
                with GradCAM(self.classifier, target_layer) as cam:
                    try:
                        heatmap, _ = cam.generate_heatmap(gradcam_tensor, class_idx=pred_idx)
                        gradcam_pil = get_gradcam_overlay(orig_pil, heatmap)
                    except Exception as e:
                        print(f"Error generating Grad-CAM: {e}")
                        traceback.print_exc()

        outlined_pil = orig_pil.resize(self.image_size)
        err_map_pil = Image.fromarray(np.zeros(self.image_size, dtype=np.uint8))
        metrics = {
            'total_area_px': 0.0,
            'total_area_cm2': 0.0,
            'width_px': 0.0,
            'width_cm': 0.0,
            'height_px': 0.0,
            'height_cm': 0.0,
            'has_tumor': False,
        }

        if pred_name != 'notumor' and self.autoencoder is not None:
            ae_tensor = self._prepare_tensor(
                self.ae_transform, orig_pil, self.device_outlining
            )
            with torch.no_grad(), _get_autocast_ctx(self.device_outlining):
                recon, att = self.autoencoder(ae_tensor)

            in_np = ae_tensor.squeeze(0).squeeze(0).float().cpu().numpy()
            recon_np = recon.squeeze(0).squeeze(0).float().cpu().numpy()
            rec_err = np.abs(in_np - recon_np)
            err_norm = (rec_err - rec_err.min()) / (rec_err.max() - rec_err.min() + 1e-8)
            err_map_pil = Image.fromarray((err_norm * 255).astype(np.uint8)).convert('L')

            _, outlined_pil, metrics = outline_tumor_enhanced(
                orig_pil,
                ae_tensor.squeeze(0),
                recon.squeeze(0),
                att.squeeze(0),
                self.image_size,
                err_thresh,
                min_solidity,
                show_plots,
                pixel_spacing_mm=pixel_spacing_mm,
            )

        return {
            'pred_name': pred_name,
            'confidence': conf_val,
            'probs': prob_dict,
            'outlined_pil': outlined_pil,
            'gradcam_pil': gradcam_pil,
            'err_map_pil': err_map_pil,
            'metrics': metrics,
        }
