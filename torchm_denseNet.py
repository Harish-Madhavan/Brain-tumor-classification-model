# --- IMPORTS ---
import os
# Thread caps BEFORE torch/numpy/sklearn load: OpenBLAS reads these at import
# time. Without this, main process + spawned DataLoader workers oversubscribe
# threads on shared-RAM Intel iGPUs -> "OpenBLAS: Memory allocation failed".
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pandas as pd
import json
from multiprocessing import freeze_support

# --- Sub-module Imports ---
from models import DenseNet121TumorClassificationModel, EnhancedAutoencoder
from dataset import advanced_preprocess, to_float_tensor, MRIDataset, NotumorMRIDataset, stratified_train_val_test_split, default_loader_kwargs, seed_everything
from database import log_diagnostic_run
from explainability import GradCAM, get_gradcam_overlay
from localization import outline_tumor_enhanced
from reporting import generate_diagnostic_report
from train import calculate_mean_std, train_classifier, train_enhanced_autoencoder, evaluate_classifier_on_test_set, compute_class_weights, plot_training_history
from app import launch_gradio_app

# --- Helper Functions (built-in torch.xpu backend, no IPEX) ---
def get_device():
    # Preferred order for this machine: Intel XPU (Arc/iGPU) -> CUDA -> CPU.
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

# --- Path Fallback Helper ---
def resolve_model_path(custom_path, default_filename):
    if custom_path and os.path.exists(custom_path):
        return custom_path
        
    res_path = os.path.join('classification_results', default_filename)
    if os.path.exists(res_path):
        return res_path
        
    # Check in workspace root
    root_path = default_filename
    if os.path.exists(root_path):
        return root_path
        
    # Try alternate filenames if default_filename is for AE
    if 'ae' in default_filename.lower() or 'autoencoder' in default_filename.lower():
        alternatives = [
            'enhanced_ae_notumor_best.pth',
            'enhanced_autoencoder_notumor_cbam_ssim_v2_best.pth',
            'enhanced_autoencoder_notumor_cbam_best.pth',
            'enhanced_autoencoder_notumor.pth'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                return alt
            alt_res = os.path.join('classification_results', alt)
            if os.path.exists(alt_res):
                return alt_res
                
    # Try alternate filenames if default_filename is for classifier
    if 'densenet' in default_filename.lower() or 'cbam' in default_filename.lower():
        alternatives = [
            'densenet121_cbam_best.pth'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                return alt
            alt_res = os.path.join('classification_results', alt)
            if os.path.exists(alt_res):
                return alt_res
                
    return res_path

# --- Model loading helpers (avoids reloading weights per image in batch mode) ---
def _infer_channels_last(device):
    return getattr(device, 'type', None) in ('xpu', 'cuda')


def load_classifier(c_path, device_classifier):
    clf = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
    clf.load_state_dict(torch.load(c_path, map_location=device_classifier))
    clf.to(device_classifier)
    if _infer_channels_last(device_classifier):
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
        ae.load_state_dict(torch.load(ae_path, map_location=device_outlining))
        ae.to(device_outlining)
        if _infer_channels_last(device_outlining):
            try:
                ae = ae.to(memory_format=torch.channels_last)
            except Exception:
                pass
        ae.eval()
        return ae
    except Exception as e:
        print(f"Warning: Failed to load autoencoder weights: {e}")
        return None


# --- Prediction Function with Saving ---
def predict_and_outline_simple(input_image_path, class_model_path, ae_model_path, image_size, norm_transform,
                            device_classifier, device_outlining, save_final_image_path=None, return_gradcam=False,
                            save_gradcam_path=None, export_report=False, report_output_path=None,
                            patient_name=None, patient_id=None, age=None, gender=None, physician_notes=None,
                            _classifier=None, _autoencoder=None, _ae_loaded=None, **kwargs):
    if not os.path.exists(input_image_path):
        print(f"Error: Not found {input_image_path}")
        return None, None, None, None, None
        
    orig_pil = Image.open(input_image_path).convert('L')
    classify_tensor = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(advanced_preprocess),
        transforms.ToTensor(),
        transforms.Lambda(to_float_tensor),
        norm_transform
    ])(orig_pil).unsqueeze(0).to(device_classifier)
    if _infer_channels_last(device_classifier):
        try:
            classify_tensor = classify_tensor.to(memory_format=torch.channels_last)
        except Exception:
            pass

    classifier = _classifier if _classifier is not None else load_classifier(class_model_path, device_classifier)
    class_map = {0:'glioma', 1:'meningioma', 2:'notumor', 3:'pituitary'}

    # Built-in XPU/CUDA autocast for faster single-image inference (bf16 XPU, fp16 CUDA).
    try:
        from train import _autocast_ctx as _infer_autocast
        _infer_ctx = _infer_autocast(device_classifier, enabled=getattr(device_classifier, 'type', None) in ('xpu', 'cuda'))
    except Exception:
        from contextlib import nullcontext as _nc
        _infer_ctx = _nc()

    with torch.no_grad(), _infer_ctx:
        pred = classifier(classify_tensor)
        probs = torch.softmax(pred.float(), 1)
        conf, idx = torch.max(probs, 1)
        pred_name = class_map[idx.item()]
        conf = conf.item()
        
    print(f"Prediction: {pred_name} (Confidence: {conf:.2%})")
    
    # Generate Grad-CAM if requested
    gradcam_pil = None
    if return_gradcam or export_report:
        classifier.eval()
        with torch.enable_grad():
            gradcam_tensor = classify_tensor.clone().detach()
            target_layer = classifier.densenet121.features.norm5
            with GradCAM(classifier, target_layer) as cam:
                try:
                    heatmap, _ = cam.generate_heatmap(gradcam_tensor, class_idx=idx.item())
                    gradcam_pil = get_gradcam_overlay(orig_pil, heatmap)
                    if return_gradcam and save_gradcam_path:
                        gradcam_pil.save(save_gradcam_path)
                        print(f"Grad-CAM overlay saved to {save_gradcam_path}")
                except Exception as e:
                    print(f"Error generating Grad-CAM: {e}")
                    traceback.print_exc()
                
    outlined_pil = orig_pil.resize(image_size)
    metrics = {
        'total_area_px': 0.0,
        'total_area_cm2': 0.0,
        'width_px': 0.0,
        'width_cm': 0.0,
        'height_px': 0.0,
        'height_cm': 0.0,
        'has_tumor': False
    }
    
    if pred_name != 'notumor':
        # Reuse preloaded autoencoder in batch mode; lazy-load otherwise.
        if _ae_loaded is not None:
            autoencoder = _autoencoder
        elif _autoencoder is not None:
            autoencoder = _autoencoder
        else:
            autoencoder = load_autoencoder(ae_model_path, device_outlining)
        if autoencoder is None:
            print("Warning: autoencoder unavailable, skipping localization.")
            autoencoder = None
    else:
        autoencoder = None

    if autoencoder is not None:
        ae_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor)
        ])(orig_pil).unsqueeze(0).to(device_outlining)
        if _infer_channels_last(device_outlining):
            try:
                ae_tensor = ae_tensor.to(memory_format=torch.channels_last)
            except Exception:
                pass

        try:
            from train import _autocast_ctx as _ae_autocast
            _ae_ctx = _ae_autocast(device_outlining, enabled=getattr(device_outlining, 'type', None) in ('xpu', 'cuda'))
        except Exception:
            from contextlib import nullcontext as _nc2
            _ae_ctx = _nc2()
        with torch.no_grad(), _ae_ctx:
            recon, att = autoencoder(ae_tensor)
        
        # Extract arguments for outline_tumor_enhanced
        err_thresh = kwargs.get('error_threshold', None)
        solidity = kwargs.get('min_solidity', 0.6)
        show_plots_outline = kwargs.get('show_plots', False)

        _, outlined_pil, metrics = outline_tumor_enhanced(
            orig_pil, ae_tensor.squeeze(0), recon.squeeze(0), att.squeeze(0),
            image_size, err_thresh, solidity, show_plots_outline
        )
    
    if save_final_image_path and outlined_pil:
        outlined_pil.save(save_final_image_path)
        print(f"Final prediction image saved to {save_final_image_path}")
        
    if kwargs.get('show_final_plot', True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(orig_pil.resize(image_size), cmap='gray')
        ax1.set_title("Original")
        ax1.axis('off')
        ax2.imshow(outlined_pil)
        ax2.set_title(f"Result: {pred_name.capitalize()} ({conf:.2%})")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    if export_report:
        if report_output_path is None:
            report_output_path = os.path.join(
                os.path.dirname(save_final_image_path) if save_final_image_path else 'classification_results',
                "diagnostic_report.html"
            )
        generate_diagnostic_report(
            input_image_path, pred_name, conf, metrics,
            orig_pil.resize(image_size), outlined_pil, gradcam_pil,
            report_output_path, str(device_classifier),
            patient_name=patient_name, patient_id=patient_id, age=age, gender=gender, physician_notes=physician_notes
        )
        
    log_diagnostic_run(
        os.path.basename(input_image_path), pred_name, conf, metrics, report_output_path if export_report else None,
        patient_name=patient_name, patient_id=patient_id, age=age, gender=gender, physician_notes=physician_notes
    )
        
    return pred_name, conf, outlined_pil, gradcam_pil, metrics

# --- Batch Inference ---
def run_batch_inference(input_dir, output_dir, class_model_path, ae_model_path, image_size, norm_transform, device_classifier, device_outlining, return_gradcam=True):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return None
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Running Batch Inference on {input_dir} ---")
    results = []

    supported_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)])

    if not image_files:
        print("No supported images found in the input directory.")
        return None

    # Load models once (major speedup vs per-image reload)
    classifier = load_classifier(class_model_path, device_classifier)
    autoencoder = load_autoencoder(ae_model_path, device_outlining)

    for filename in tqdm(image_files, desc="Batch Inference Progress"):
        img_path = os.path.join(input_dir, filename)
        save_img_name = f"pred_{filename}"
        save_img_path = os.path.join(output_dir, save_img_name)
        
        save_gradcam_name = f"gradcam_{filename}"
        save_gradcam_path = os.path.join(output_dir, save_gradcam_name) if return_gradcam else None
        
        try:
            pred_name, conf, outlined_pil, gradcam_pil, metrics = predict_and_outline_simple(
                img_path, class_model_path, ae_model_path, image_size, norm_transform,
                device_classifier, device_outlining, save_final_image_path=save_img_path,
                return_gradcam=return_gradcam, save_gradcam_path=save_gradcam_path, show_final_plot=False,
                _classifier=classifier, _autoencoder=autoencoder, _ae_loaded=True
            )
            
            results.append({
                'filename': filename,
                'predicted_class': pred_name,
                'confidence': conf,
                'tumor_area_cm2': metrics['total_area_cm2'] if metrics else 0.0,
                'tumor_width_cm': metrics['width_cm'] if metrics else 0.0,
                'tumor_height_cm': metrics['height_cm'] if metrics else 0.0,
                'output_image_path': save_img_path,
                'gradcam_image_path': save_gradcam_path if (return_gradcam and gradcam_pil is not None) else 'N/A'
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save results to CSV
    report_path = os.path.join(output_dir, "batch_inference_report.csv")
    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False)
    print(f"\nBatch inference complete. Report saved to: {report_path}")
    return report_path

# --- Main Execution ---
def _cap_host_threads_for_xpu():
    # Prevent OpenBLAS/MKL thread explosion: each DataLoader worker + the
    # main process default to many threads -> oversubscription + the
    # "OpenBLAS: Memory allocation failed after 10 retries" abort seen on
    # shared-RAM Intel iGPUs. Must run before any DataLoader is created.
    try:
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass


def main():
    import argparse
    freeze_support()
    _cap_host_threads_for_xpu()
    
    parser = argparse.ArgumentParser(description="Brain Tumor Classification & Localization Model Pipeline")
    parser.add_argument('--mode', type=str, default='app', choices=['app', 'predict', 'batch', 'train', 'evaluate'],
                        help="Execution mode: 'app' (Gradio web GUI), 'predict' (single-image prediction), 'batch' (batch processing), 'train' (model training), 'evaluate' (test-set evaluation)")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Path to training data directory (contains glioma, meningioma, etc.)")
    parser.add_argument('--results_dir', type=str, default='classification_results',
                        help="Output directory for reports, plots, and weights")
    parser.add_argument('--c_path', type=str, default=None,
                        help="Path to classifier model weights (.pth)")
    parser.add_argument('--ae_path', type=str, default=None,
                        help="Path to autoencoder model weights (.pth)")
    parser.add_argument('--image', type=str, default=None,
                        help="Path to input image for single-image prediction")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Input directory of images for batch inference")
    parser.add_argument('--output_dir', type=str, default='batch_output',
                        help="Output directory for batch inference results")
    parser.add_argument('--gradcam', action='store_true',
                        help="Generate Grad-CAM activation map in predict/batch mode")
    parser.add_argument('--export_report', action='store_true',
                        help="Export printable HTML diagnostic report in predict mode")
    parser.add_argument('--patient_name', type=str, default=None,
                        help="Patient name for diagnostic reporting")
    parser.add_argument('--patient_id', type=str, default=None,
                        help="Patient ID code for diagnostic reporting")
    parser.add_argument('--age', type=int, default=None,
                        help="Patient age for diagnostic reporting")
    parser.add_argument('--gender', type=str, default=None, choices=['Male', 'Female', 'Other'],
                        help="Patient gender for diagnostic reporting")
    parser.add_argument('--notes', type=str, default=None,
                        help="Clinical radiologist notes / physician comments")
    parser.add_argument('--epochs', type=int, default=15,
                        help="Classifier training epochs (default 15)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Training batch size (default 32; larger=faster, needs more RAM)")
    parser.add_argument('--workers', type=int, default=None,
                        help="DataLoader workers (default: 0 on XPU/Windows for RAM safety, else auto)")
    parser.add_argument('--no-amp', action='store_true',
                        help="Disable mixed-precision training")
    parser.add_argument('--amp-dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16'],
                        help="AMP dtype: auto=bf16 on XPU/CPU, fp16 on CUDA (default auto)")
    parser.add_argument('--no-channels-last', action='store_true',
                        help="Disable channels_last memory format (enabled by default on XPU/CUDA)")
    parser.add_argument('--mixup', type=float, default=0.0,
                        help="MixUp alpha for regularization, e.g. 0.2 (default 0 = off)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for splits, shuffling and augmentation (default 42)")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    # Resolve paths
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Resolve model paths using fallbacks
    c_path = resolve_model_path(args.c_path, 'densenet121_cbam_best.pth')
    ae_path = resolve_model_path(args.ae_path, 'enhanced_ae_notumor_best.pth')
    
    print(f"Classifier weights resolved to: {c_path}")
    print(f"Autoencoder weights resolved to: {ae_path}")
    
    image_size = (256, 256)
    
    # Resolve dataset directory (no machine-specific absolute fallbacks)
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = 'training1'

    print(f"Data directory: {data_dir}")

    device = get_device()

    # Compute mean and std for normalization (train split only to avoid leakage)
    precomputed_mean = torch.tensor([0.1834])
    precomputed_std = torch.tensor([0.1982])

    mean, std = precomputed_mean, precomputed_std
    if os.path.exists(data_dir):
        try:
            full_ds_for_stats = MRIDataset(data_dir, image_size, 'full', False)
            if len(full_ds_for_stats) > 0:
                tr_idx, _, _ = stratified_train_val_test_split(
                    full_ds_for_stats.labels, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
                if tr_idx:
                    _stat_kw = default_loader_kwargs(batch_size=64, num_workers=args.workers, device=device)
                    _stat_kw.pop('batch_size', None)
                    mean, std = calculate_mean_std(DataLoader(torch.utils.data.Subset(full_ds_for_stats, tr_idx), batch_size=64, shuffle=False, **_stat_kw))
                    print(f"Calculated dataset stats: Mean={mean.item():.4f}, Std={std.item():.4f}")
        except Exception as e:
            print(f"Failed to calculate stats, using precomputed defaults: {e}")
            
    norm = transforms.Normalize(mean=mean, std=std)
    
    if args.mode == 'app':
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}. Please train the model first or provide the correct path with --c_path.")
            return
        launch_gradio_app(c_path, ae_path, image_size, norm, device, torch.device('cpu'))
        
    elif args.mode == 'predict':
        sample_image_path = args.image
        if not sample_image_path:
            # Deterministic fallback: first image (sorted) under data_dir, no user-specific paths.
            found_sample = False
            if os.path.exists(data_dir):
                for folder in sorted(os.listdir(data_dir)):
                    folder_path = os.path.join(data_dir, folder)
                    if os.path.isdir(folder_path):
                        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        if files:
                            sample_image_path = os.path.join(folder_path, files[0])
                            found_sample = True
                            break
            if not found_sample:
                print("Error: Please specify an image using --image <path>.")
                return
                
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return
            
        print(f"\n--- Performing Prediction on {sample_image_path} ---")
        save_img_path = os.path.join(results_dir, "prediction_outlined.png")
        save_gradcam_path = os.path.join(results_dir, "prediction_gradcam.png") if args.gradcam else None
        
        predict_and_outline_simple(
            sample_image_path, c_path, ae_path, image_size, norm, device, torch.device('cpu'),
            save_final_image_path=save_img_path, return_gradcam=args.gradcam, save_gradcam_path=save_gradcam_path,
            export_report=args.export_report, error_threshold=None, min_solidity=0.65, show_plots=False, show_final_plot=True,
            patient_name=args.patient_name, patient_id=args.patient_id, age=args.age, gender=args.gender, physician_notes=args.notes
        )
        
    elif args.mode == 'batch':
        input_dir = args.input_dir
        if not input_dir:
            print("Error: Please specify input directory of images using --input_dir <path>.")
            return
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return
        run_batch_inference(
            input_dir, args.output_dir, c_path, ae_path, image_size, norm, device, torch.device('cpu'),
            return_gradcam=args.gradcam
        )
        
    elif args.mode == 'train':
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist. Cannot train.")
            return
            
        print("\n--- Model Training Mode ---")
        batch_size = args.batch_size
        use_amp = not args.no_amp
        amp_dtype = None if args.amp_dtype == 'auto' else args.amp_dtype
        use_cl = not args.no_channels_last
        try:
            from train import _amp_dtype_for as _resolve_amp_dtype
            print(f"Backend={device.type} AMP={use_amp} dtype={_resolve_amp_dtype(device, override=amp_dtype) if use_amp else 'fp32'} channels_last={use_cl}")
        except Exception:
            pass
        # Stratified 70/15/15 split — train/val/test are now disjoint.
        probe_ds = MRIDataset(data_dir, image_size, 'full', apply_normalize=False)
        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            probe_ds.labels, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
        print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (held-out)")
        if not train_idx or not val_idx:
            print("Error: not enough data for a stratified split.")
            return
        loader_kw = default_loader_kwargs(batch_size=batch_size, num_workers=args.workers, device=device)
        print(f"Loader: batch={batch_size} workers={loader_kw.get('num_workers')} pin_memory={loader_kw.get('pin_memory', False)} AMP={use_amp}")
        # Compute mean/std from train split only (unnormalized stats loader)
        train_loader_for_stats = DataLoader(
            MRIDataset(data_dir, image_size, 'full', False, indices=train_idx),
            batch_size=64, shuffle=False, num_workers=loader_kw.get('num_workers', 0))
        m, s = calculate_mean_std(train_loader_for_stats)
        norm = transforms.Normalize(mean=m, std=s)
        train_dataset = MRIDataset(data_dir, image_size, 'train', apply_normalize=True, indices=train_idx)
        train_dataset.set_normalization(norm)

        # Load validation (no augmentation)
        val_dataset = MRIDataset(data_dir, image_size, 'val', apply_normalize=True, indices=val_idx)
        val_dataset.set_normalization(norm)

        train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

        classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=True)
        classifier.to(device)

        # Accuracy: class-weighted + label-smoothed CE (notumor is ~28% of data)
        train_labels = [probe_ds.labels[i] for i in train_idx]
        class_w = compute_class_weights(train_labels, num_classes=4, device=device)
        print(f"Class weights: {[round(float(w), 3) for w in class_w]}")
        try:
            criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=class_w)

        # Speed + accuracy: 2-epoch frozen-backbone head warmup, then differential-LR fine-tune
        classifier.freeze_backbone()
        warm_opt = torch.optim.AdamW(classifier.head_params(), lr=1e-3, weight_decay=1e-4)
        warm_sched = CosineAnnealingLR(warm_opt, T_max=max(1, args.epochs))
        print("Phase 1/2: head-only warmup (2 epochs, backbone frozen)...")
        train_classifier(classifier, train_loader, val_loader, criterion, warm_opt, warm_sched, min(2, args.epochs),
                         device, c_path, use_amp=use_amp, mixup_alpha=0.0, amp_dtype=amp_dtype, channels_last=use_cl)

        classifier.unfreeze_all()
        # Reload best warmup checkpoint before full fine-tune if it exists
        if os.path.exists(c_path):
            try:
                classifier.load_state_dict(torch.load(c_path, map_location=device))
            except Exception as e:
                print(f"Warning: could not reload warmup checkpoint: {e}")
        ft_opt = torch.optim.AdamW([
            {'params': classifier.backbone_params(), 'lr': 3e-5},
            {'params': classifier.head_params(), 'lr': 1e-4},
        ], weight_decay=1e-4)
        ft_sched = CosineAnnealingLR(ft_opt, T_max=max(1, args.epochs))
        print("Phase 2/2: full fine-tune with differential LR + cosine schedule...")
        history = train_classifier(classifier, train_loader, val_loader, criterion, ft_opt, ft_sched, args.epochs,
                                   device, c_path, use_amp=use_amp, mixup_alpha=args.mixup, amp_dtype=amp_dtype, channels_last=use_cl)

        # Persist training artifacts (history was previously only kept in memory)
        try:
            plot_training_history(history, save_path=os.path.join(results_dir, "training_history.png"))
            with open(os.path.join(results_dir, "training_history.json"), 'w') as _hf:
                json.dump({'history': history,
                           'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
                           'seed': args.seed}, _hf, indent=2)
        except Exception as e:
            print(f"Warning: could not save training history: {e}")

        # Train Autoencoder on healthy scans from TRAIN split only (avoids val/test leakage)
        # XPU/iGPU note: SSIM + 256px reconstructions are RAM-hungry on shared
        # memory, so cap the AE batch below the classifier batch on XPU.
        print("\nTraining tumor-outlining Autoencoder...")
        full_nt = NotumorMRIDataset(data_dir, image_size, 'full')
        # Map global train indices -> notumor-only subset by matching file paths
        train_files = {probe_ds.image_paths[i] for i in train_idx}
        nt_train_idx = [i for i, p in enumerate(full_nt.image_paths) if p in train_files]
        if not nt_train_idx:
            # Fallback: stratified split of notumor pool alone
            from sklearn.model_selection import train_test_split as _tts
            nt_train_idx, _ = _tts(list(range(len(full_nt))), test_size=0.3, random_state=args.seed)
        ae_batch = min(batch_size, 16) if getattr(device, 'type', None) == 'xpu' else batch_size
        ae_kw = default_loader_kwargs(batch_size=ae_batch, num_workers=args.workers, device=device)
        print(f"AE loader: batch={ae_batch} workers={ae_kw.get('num_workers')}")
        ae_train_dataset = NotumorMRIDataset(data_dir, image_size, 'train', indices=nt_train_idx)
        ae_train_loader = DataLoader(ae_train_dataset, shuffle=True, **ae_kw)
        autoencoder = EnhancedAutoencoder()
        autoencoder.to(device)
        try:
            train_enhanced_autoencoder(autoencoder, ae_train_loader, 20, device, ae_path, 0.1, 0.4, use_amp=use_amp, amp_dtype=amp_dtype, channels_last=use_cl)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            print(f"AE OOM at batch={ae_batch}, retrying at batch={ae_batch // 2} workers=0...")
            try:
                if hasattr(torch, 'xpu'):
                    torch.xpu.empty_cache()
            except Exception:
                pass
            ae_kw = default_loader_kwargs(batch_size=max(4, ae_batch // 2), num_workers=0, device=device)
            ae_train_loader = DataLoader(ae_train_dataset, shuffle=True, **ae_kw)
            train_enhanced_autoencoder(autoencoder, ae_train_loader, 20, device, ae_path, 0.1, 0.4, use_amp=use_amp, amp_dtype=amp_dtype, channels_last=use_cl)

    elif args.mode == 'evaluate':
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist. Cannot evaluate.")
            return
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return

        probe_ds = MRIDataset(data_dir, image_size, 'full', apply_normalize=False)
        _, _, test_idx = stratified_train_val_test_split(
            probe_ds.labels, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
        if not test_idx:
            print("Error: test split is empty.")
            return
        print(f"Evaluating on held-out test split: {len(test_idx)} images")
        test_dataset = MRIDataset(data_dir, image_size, 'test', apply_normalize=True, indices=test_idx)
        test_dataset.set_normalization(norm)
        test_loader = DataLoader(test_dataset, shuffle=False, **default_loader_kwargs(batch_size=64, num_workers=args.workers, device=device))
        
        classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
        classifier.load_state_dict(torch.load(c_path, map_location=device))
        classifier.to(device)
        
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        evaluate_classifier_on_test_set(classifier, test_loader, device, class_names, mean, std, results_dir)

if __name__ == '__main__':
    main()