# --- IMPORTS ---
import os
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from tqdm import tqdm
import pandas as pd
from multiprocessing import freeze_support

# --- Sub-module Imports ---
from models import DenseNet121TumorClassificationModel, EnhancedAutoencoder
from dataset import equalize_histogram, advanced_preprocess, to_float_tensor, MRIDataset, NotumorMRIDataset
from database import init_db, log_diagnostic_run, fetch_diagnostic_records, delete_diagnostic_record, export_database_csv, get_database_stats
from explainability import GradCAM, get_gradcam_overlay
from localization import enhance_outline_detection, outline_tumor_enhanced
from reporting import generate_diagnostic_report
from train import calculate_mean_std, train_classifier, train_enhanced_autoencoder, evaluate_classifier_on_test_set
from app import launch_gradio_app

# --- Helper Functions ---
def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("Using XPU device.")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("XPU not available, using CUDA device.")
        return torch.device("cuda")
    else:
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

# --- Prediction Function with Saving ---
def predict_and_outline_simple(input_image_path, class_model_path, ae_model_path, image_size, norm_transform,
                            device_classifier, device_outlining, save_final_image_path=None, return_gradcam=False,
                            save_gradcam_path=None, export_report=False, report_output_path=None,
                            patient_name=None, patient_id=None, age=None, gender=None, physician_notes=None, **kwargs):
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
    
    classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
    classifier.load_state_dict(torch.load(class_model_path, map_location=device_classifier))
    classifier.to(device_classifier).eval()
    class_map = {0:'glioma', 1:'meningioma', 2:'notumor', 3:'pituitary'}
    
    with torch.no_grad():
        pred = classifier(classify_tensor)
        probs = torch.softmax(pred, 1)
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
    
    if pred_name != 'notumor' and os.path.exists(ae_model_path):
        ae_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor)
        ])(orig_pil).unsqueeze(0).to(device_outlining)
        
        autoencoder = EnhancedAutoencoder()
        autoencoder.load_state_dict(torch.load(ae_model_path, map_location=device_outlining))
        autoencoder.to(device_outlining).eval()
        
        with torch.no_grad():
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
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]
    
    if not image_files:
        print("No supported images found in the input directory.")
        return None
        
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
                return_gradcam=return_gradcam, save_gradcam_path=save_gradcam_path, show_final_plot=False
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
def main():
    import argparse
    freeze_support()
    
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
    
    args = parser.parse_args()
    
    # Resolve paths
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Resolve model paths using fallbacks
    c_path = resolve_model_path(args.c_path, 'densenet121_cbam_best.pth')
    ae_path = resolve_model_path(args.ae_path, 'enhanced_ae_notumor_best.pth')
    
    print(f"Classifier weights resolved to: {c_path}")
    print(f"Autoencoder weights resolved to: {ae_path}")
    
    image_size = (256, 256)
    
    # Resolve dataset directory
    data_dir = args.data_dir
    if data_dir is None:
        if os.path.exists('training1'):
            data_dir = 'training1'
        elif os.path.exists('E:\\project\\dataset1\\training1'):
            data_dir = 'E:\\project\\dataset1\\training1'
        else:
            data_dir = 'training1'
            
    print(f"Data directory: {data_dir}")
    
    device = get_device()
    
    # Compute mean and std for normalization
    precomputed_mean = torch.tensor([0.1834])
    precomputed_std = torch.tensor([0.1982])
    
    mean, std = precomputed_mean, precomputed_std
    if os.path.exists(data_dir):
        try:
            full_ds_for_stats = MRIDataset(data_dir, image_size, 'full', False)
            if len(full_ds_for_stats) > 0:
                labels = full_ds_for_stats.labels
                train_idx, _ = train_test_split(range(len(full_ds_for_stats)), test_size=0.3, random_state=42, stratify=labels if min(Counter(labels).values()) > 1 else None)
                mean, std = calculate_mean_std(DataLoader(torch.utils.data.Subset(full_ds_for_stats, train_idx), 16, num_workers=0))
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
            sample_image_path = "C:\\Users\\Harish\\Downloads\\PituitaryAdenoma.jpeg"
            if not os.path.exists(sample_image_path):
                # Try finding a random image in data_dir if it exists
                found_sample = False
                if os.path.exists(data_dir):
                    for folder in os.listdir(data_dir):
                        folder_path = os.path.join(data_dir, folder)
                        if os.path.isdir(folder_path):
                            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
            # Try to use test directory or validation subdirectory as default fallback
            if os.path.exists(data_dir):
                input_dir = os.path.join(data_dir, 'pituitary') # default directory if none given
                if not os.path.exists(input_dir):
                    input_dir = data_dir
            else:
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
        # Initialize datasets
        train_dataset = MRIDataset(data_dir, image_size, 'train', apply_normalize=True)
        # Compute mean and std dynamically from training loader
        train_loader_for_stats = DataLoader(train_dataset, batch_size=16, shuffle=False)
        m, s = calculate_mean_std(train_loader_for_stats)
        norm = transforms.Normalize(mean=m, std=s)
        train_dataset.set_normalization(norm)
        
        # Load validation
        val_dataset = MRIDataset(data_dir, image_size, 'val', apply_normalize=True)
        val_dataset.set_normalization(norm)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=True)
        classifier.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        history = train_classifier(classifier, train_loader, val_loader, criterion, optimizer, scheduler, 15, device, c_path)
        
        # Train Autoencoder
        print("\nTraining tumor-outlining Autoencoder...")
        ae_train_dataset = NotumorMRIDataset(data_dir, image_size, 'train')
        ae_train_loader = DataLoader(ae_train_dataset, batch_size=16, shuffle=True)
        autoencoder = EnhancedAutoencoder()
        autoencoder.to(device)
        train_enhanced_autoencoder(autoencoder, ae_train_loader, 20, device, ae_path, 0.1, 0.4)
        
    elif args.mode == 'evaluate':
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist. Cannot evaluate.")
            return
        if not os.path.exists(c_path):
            print(f"Error: Classifier weights not found at {c_path}.")
            return
            
        test_dataset = MRIDataset(data_dir, image_size, 'test', apply_normalize=True)
        test_dataset.set_normalization(norm)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
        classifier.load_state_dict(torch.load(c_path, map_location=device))
        classifier.to(device)
        
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        evaluate_classifier_on_test_set(classifier, test_loader, device, class_names, mean, std, results_dir)

if __name__ == '__main__':
    main()