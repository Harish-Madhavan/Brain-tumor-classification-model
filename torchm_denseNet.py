# --- IMPORTS ---
import os
# Thread caps BEFORE torch/numpy/sklearn load: OpenBLAS reads these at import
# time. Without this, main process + spawned DataLoader workers oversubscribe
# threads on shared-RAM Intel iGPUs -> "OpenBLAS: Memory allocation failed".
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- Sub-module Imports ---
from app import launch_gradio_app
from database import log_diagnostic_run
from dataset import (
    MRIDataset,
    NotumorMRIDataset,
    default_loader_kwargs,
    seed_everything,
    stratified_train_val_test_split,
)
from inference import (
    CLASS_NAMES,
    DEFAULT_MEAN,
    DEFAULT_STD,
    DiagnosticPipeline,
    get_device,
    load_autoencoder,
    load_classifier,
    load_model_metadata,
    resolve_model_path,
    safe_torch_load,
    save_model_metadata,
)
from models import DenseNet121TumorClassificationModel, EnhancedAutoencoder
from reporting import generate_diagnostic_report
from train import (
    calculate_mean_std,
    compute_class_weights,
    evaluate_classifier_on_test_set,
    plot_training_history,
    train_classifier,
    train_enhanced_autoencoder,
)


def predict_and_outline_simple(
    input_image_path, class_model_path, ae_model_path, image_size, norm_transform,
    device_classifier, device_outlining, save_final_image_path=None, return_gradcam=False,
    save_gradcam_path=None, export_report=False, report_output_path=None,
    patient_name=None, patient_id=None, age=None, gender=None, physician_notes=None,
    _classifier=None, _autoencoder=None, _ae_loaded=None, _pipeline=None, **kwargs
):
    if not os.path.exists(input_image_path):
        print(f"Error: Not found {input_image_path}")
        return None, None, None, None, None

    if _pipeline is None:
        classifier = _classifier if _classifier is not None else load_classifier(class_model_path, device_classifier)
        if _ae_loaded is not None or _autoencoder is not None:
            autoencoder = _autoencoder
        else:
            autoencoder = load_autoencoder(ae_model_path, device_outlining)
        _pipeline = DiagnosticPipeline(
            classifier, autoencoder, image_size, norm_transform,
            device_classifier, device_outlining
        )

    orig_pil = Image.open(input_image_path).convert('L')
    result = _pipeline.predict(
        orig_pil,
        generate_gradcam=(return_gradcam or export_report),
        err_thresh=kwargs.get('error_threshold', None),
        min_solidity=kwargs.get('min_solidity', 0.6),
        pixel_spacing_mm=kwargs.get('pixel_spacing_mm', 1.15),
        show_plots=kwargs.get('show_plots', False),
    )

    pred_name = result['pred_name']
    conf = result['confidence']
    outlined_pil = result['outlined_pil']
    gradcam_pil = result['gradcam_pil']
    metrics = result['metrics']

    print(f"Prediction: {pred_name} (Confidence: {conf:.2%})")

    if return_gradcam and save_gradcam_path and gradcam_pil is not None:
        gradcam_pil.save(save_gradcam_path)
        print(f"Grad-CAM overlay saved to {save_gradcam_path}")

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


def run_batch_inference(input_dir, output_dir, class_model_path, ae_model_path, image_size, norm_transform,
                        device_classifier, device_outlining, return_gradcam=True, pixel_spacing_mm=1.15):
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

    pipeline = DiagnosticPipeline.from_paths(
        class_model_path, ae_model_path, image_size, norm_transform,
        device_classifier, device_outlining
    )

    for filename in tqdm(image_files, desc="Batch Inference Progress"):
        img_path = os.path.join(input_dir, filename)
        save_img_path = os.path.join(output_dir, f"pred_{filename}")
        save_gradcam_path = os.path.join(output_dir, f"gradcam_{filename}") if return_gradcam else None

        try:
            pred_name, conf, _, gradcam_pil, metrics = predict_and_outline_simple(
                img_path, class_model_path, ae_model_path, image_size, norm_transform,
                device_classifier, device_outlining, save_final_image_path=save_img_path,
                return_gradcam=return_gradcam, save_gradcam_path=save_gradcam_path,
                show_final_plot=False, pixel_spacing_mm=pixel_spacing_mm, _pipeline=pipeline
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

    report_path = os.path.join(output_dir, "batch_inference_report.csv")
    pd.DataFrame(results).to_csv(report_path, index=False)
    print(f"\nBatch inference complete. Report saved to: {report_path}")
    return report_path


def _cap_host_threads_for_xpu():
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


def _resolve_norm_stats(data_dir, results_dir, image_size, workers, device, seed):
    meta = load_model_metadata(results_dir)
    if meta and 'mean' in meta and 'std' in meta:
        mean = torch.tensor(meta['mean'], dtype=torch.float32)
        std = torch.tensor(meta['std'], dtype=torch.float32)
        print(f"Loaded persisted normalization stats: Mean={mean.item():.4f}, Std={std.item():.4f}")
        return mean, std

    mean = torch.tensor(DEFAULT_MEAN, dtype=torch.float32)
    std = torch.tensor(DEFAULT_STD, dtype=torch.float32)
    if os.path.exists(data_dir):
        try:
            full_ds_for_stats = MRIDataset(data_dir, image_size, 'full', False)
            if len(full_ds_for_stats) > 0:
                tr_idx, _, _ = stratified_train_val_test_split(
                    full_ds_for_stats.labels, val_ratio=0.15, test_ratio=0.15, seed=seed
                )
                if tr_idx:
                    stat_kw = default_loader_kwargs(batch_size=64, num_workers=workers, device=device)
                    stat_kw.pop('batch_size', None)
                    mean, std = calculate_mean_std(
                        DataLoader(torch.utils.data.Subset(full_ds_for_stats, tr_idx), batch_size=64, shuffle=False, **stat_kw)
                    )
                    print(f"Calculated dataset stats: Mean={mean.item():.4f}, Std={std.item():.4f}")
        except Exception as e:
            print(f"Failed to calculate stats, using precomputed defaults: {e}")
    return mean, std


def main():
    import argparse
    freeze_support()
    _cap_host_threads_for_xpu()

    parser = argparse.ArgumentParser(description="Brain Tumor Classification & Localization Model Pipeline")
    parser.add_argument('--mode', type=str, default='app', choices=['app', 'predict', 'batch', 'train', 'evaluate'],
                        help="Execution mode: 'app' (Gradio web GUI), 'predict', 'batch', 'train', or 'evaluate'")
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
    parser.add_argument('--pixel-spacing', type=float, default=1.15,
                        help="Assumed MRI pixel spacing in mm/pixel (default 1.15)")
    parser.add_argument('--device-outlining', type=str, default='auto', choices=['auto', 'cpu'],
                        help="Compute device for autoencoder localization: 'auto' (match primary device) or 'cpu'")
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

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    c_path = resolve_model_path(args.c_path, 'densenet121_cbam_best.pth')
    ae_path = resolve_model_path(args.ae_path, 'enhanced_ae_notumor_best.pth')
    print(f"Classifier weights resolved to: {c_path}")
    print(f"Autoencoder weights resolved to: {ae_path}")

    image_size = (256, 256)
    data_dir = args.data_dir if args.data_dir is not None else 'training1'
    print(f"Data directory: {data_dir}")

    device = get_device()
    outlining_device = torch.device('cpu') if args.device_outlining == 'cpu' else device

    if args.mode == 'app':
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}. Please train the model first or provide the correct path with --c_path.")
            return
        mean, std = _resolve_norm_stats(data_dir, results_dir, image_size, args.workers, device, args.seed)
        norm = transforms.Normalize(mean=mean, std=std)
        launch_gradio_app(c_path, ae_path, image_size, norm, device, outlining_device)

    elif args.mode == 'predict':
        sample_image_path = args.image
        if not sample_image_path:
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

        mean, std = _resolve_norm_stats(data_dir, results_dir, image_size, args.workers, device, args.seed)
        norm = transforms.Normalize(mean=mean, std=std)
        print(f"\n--- Performing Prediction on {sample_image_path} ---")
        save_img_path = os.path.join(results_dir, "prediction_outlined.png")
        save_gradcam_path = os.path.join(results_dir, "prediction_gradcam.png") if args.gradcam else None

        predict_and_outline_simple(
            sample_image_path, c_path, ae_path, image_size, norm, device, outlining_device,
            save_final_image_path=save_img_path, return_gradcam=args.gradcam, save_gradcam_path=save_gradcam_path,
            export_report=args.export_report, error_threshold=None, min_solidity=0.65,
            pixel_spacing_mm=args.pixel_spacing, show_plots=False, show_final_plot=True,
            patient_name=args.patient_name, patient_id=args.patient_id, age=args.age, gender=args.gender, physician_notes=args.notes
        )

    elif args.mode == 'batch':
        if not args.input_dir:
            print("Error: Please specify input directory of images using --input_dir <path>.")
            return
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return
        mean, std = _resolve_norm_stats(data_dir, results_dir, image_size, args.workers, device, args.seed)
        norm = transforms.Normalize(mean=mean, std=std)
        run_batch_inference(
            args.input_dir, args.output_dir, c_path, ae_path, image_size, norm, device, outlining_device,
            return_gradcam=args.gradcam, pixel_spacing_mm=args.pixel_spacing
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

        probe_ds = MRIDataset(data_dir, image_size, 'full', apply_normalize=False)
        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            probe_ds.labels, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
        print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (held-out)")
        if not train_idx or not val_idx:
            print("Error: not enough data for a stratified split.")
            return

        loader_kw = default_loader_kwargs(batch_size=batch_size, num_workers=args.workers, device=device)
        print(f"Loader: batch={batch_size} workers={loader_kw.get('num_workers')} pin_memory={loader_kw.get('pin_memory', False)} AMP={use_amp}")

        train_loader_for_stats = DataLoader(
            MRIDataset(data_dir, image_size, 'full', False, indices=train_idx),
            batch_size=64, shuffle=False, num_workers=loader_kw.get('num_workers', 0))
        m, s = calculate_mean_std(train_loader_for_stats)
        save_model_metadata(results_dir, m, s, image_size=image_size, class_names=CLASS_NAMES)
        norm = transforms.Normalize(mean=m, std=s)

        train_dataset = MRIDataset(data_dir, image_size, 'train', apply_normalize=True, indices=train_idx)
        train_dataset.set_normalization(norm)
        val_dataset = MRIDataset(data_dir, image_size, 'val', apply_normalize=True, indices=val_idx)
        val_dataset.set_normalization(norm)

        train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

        classifier = DenseNet121TumorClassificationModel(num_classes=len(CLASS_NAMES), pretrained=True)
        classifier.to(device)

        train_labels = [probe_ds.labels[i] for i in train_idx]
        class_w = compute_class_weights(train_labels, num_classes=len(CLASS_NAMES), device=device)
        print(f"Class weights: {[round(float(w), 3) for w in class_w]}")
        try:
            criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=class_w)

        warmup_epochs = min(2, args.epochs)
        classifier.freeze_backbone()
        warm_opt = torch.optim.AdamW(classifier.head_params(), lr=1e-3, weight_decay=1e-4)
        warm_sched = CosineAnnealingLR(warm_opt, T_max=max(1, warmup_epochs))
        print(f"Phase 1/2: head-only warmup ({warmup_epochs} epochs, backbone frozen)...")
        train_classifier(classifier, train_loader, val_loader, criterion, warm_opt, warm_sched, warmup_epochs,
                         device, c_path, use_amp=use_amp, mixup_alpha=0.0, amp_dtype=amp_dtype, channels_last=use_cl)

        classifier.unfreeze_all()
        if os.path.exists(c_path):
            try:
                classifier.load_state_dict(safe_torch_load(c_path, map_location=device))
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

        try:
            plot_training_history(history, save_path=os.path.join(results_dir, "training_history.png"))
            with open(os.path.join(results_dir, "training_history.json"), 'w', encoding='utf-8') as _hf:
                json.dump({'history': history,
                           'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
                           'seed': args.seed}, _hf, indent=2)
        except Exception as e:
            print(f"Warning: could not save training history: {e}")

        print("\nTraining tumor-outlining Autoencoder...")
        full_nt = NotumorMRIDataset(data_dir, image_size, 'full')
        train_files = {probe_ds.image_paths[i] for i in train_idx}
        val_files = {probe_ds.image_paths[i] for i in val_idx}
        nt_train_idx = [i for i, p in enumerate(full_nt.image_paths) if p in train_files]
        nt_val_idx = [i for i, p in enumerate(full_nt.image_paths) if p in val_files]
        if not nt_train_idx:
            from sklearn.model_selection import train_test_split as _tts
            nt_train_idx, nt_val_idx = _tts(list(range(len(full_nt))), test_size=0.2, random_state=args.seed)

        ae_batch = min(batch_size, 16) if getattr(device, 'type', None) == 'xpu' else batch_size
        ae_kw = default_loader_kwargs(batch_size=ae_batch, num_workers=args.workers, device=device)
        print(f"AE loader: batch={ae_batch} workers={ae_kw.get('num_workers')}")
        ae_train_dataset = NotumorMRIDataset(data_dir, image_size, 'train', indices=nt_train_idx)
        ae_train_loader = DataLoader(ae_train_dataset, shuffle=True, **ae_kw)
        ae_val_loader = (
            DataLoader(NotumorMRIDataset(data_dir, image_size, 'val', indices=nt_val_idx), shuffle=False, **ae_kw)
            if nt_val_idx else None
        )
        autoencoder = EnhancedAutoencoder()
        autoencoder.to(device)
        try:
            train_enhanced_autoencoder(
                autoencoder, ae_train_loader, 20, device, ae_path, 0.1, 0.4,
                use_amp=use_amp, amp_dtype=amp_dtype, channels_last=use_cl, val_loader=ae_val_loader
            )
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
            ae_val_loader = (
                DataLoader(NotumorMRIDataset(data_dir, image_size, 'val', indices=nt_val_idx), shuffle=False, **ae_kw)
                if nt_val_idx else None
            )
            train_enhanced_autoencoder(
                autoencoder, ae_train_loader, 20, device, ae_path, 0.1, 0.4,
                use_amp=use_amp, amp_dtype=amp_dtype, channels_last=use_cl, val_loader=ae_val_loader
            )

    elif args.mode == 'evaluate':
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist. Cannot evaluate.")
            return
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return

        mean, std = _resolve_norm_stats(data_dir, results_dir, image_size, args.workers, device, args.seed)
        norm = transforms.Normalize(mean=mean, std=std)

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

        classifier = load_classifier(c_path, device)
        amp_dtype = None if args.amp_dtype == 'auto' else args.amp_dtype
        evaluate_classifier_on_test_set(
            classifier, test_loader, device, CLASS_NAMES, mean, std, results_dir,
            use_amp=not args.no_amp, amp_dtype=amp_dtype, channels_last=not args.no_channels_last
        )


if __name__ == '__main__':
    main()
