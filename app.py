import os
import uuid
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import gradio as gr

from models import DenseNet121TumorClassificationModel, EnhancedAutoencoder
from dataset import advanced_preprocess, to_float_tensor
from explainability import GradCAM, get_gradcam_overlay
from localization import enhance_outline_detection, outline_tumor_enhanced
from database import fetch_diagnostic_records, get_database_stats, delete_diagnostic_record, export_database_csv, log_diagnostic_run
from reporting import generate_diagnostic_report

def launch_gradio_app(class_model_path, ae_model_path, image_size, norm_transform, device_classifier, device_outlining):
    # Load models once to keep web app fast (channels_last helps Intel XPU/CUDA convs)
    classifier = DenseNet121TumorClassificationModel(num_classes=4, pretrained=False)
    classifier.load_state_dict(torch.load(class_model_path, map_location=device_classifier))
    classifier.to(device_classifier)
    if getattr(device_classifier, 'type', None) in ('xpu', 'cuda'):
        try:
            classifier = classifier.to(memory_format=torch.channels_last)
        except Exception:
            pass
    classifier.eval()

    autoencoder = None
    if os.path.exists(ae_model_path):
        try:
            autoencoder = EnhancedAutoencoder()
            autoencoder.load_state_dict(torch.load(ae_model_path, map_location=device_outlining))
            autoencoder.to(device_outlining)
            if getattr(device_outlining, 'type', None) in ('xpu', 'cuda'):
                try:
                    autoencoder = autoencoder.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            autoencoder.eval()
            print("Autoencoder weights loaded successfully for localization.")
        except Exception as e:
            print(f"Warning: Failed to load autoencoder weights: {e}")
        
    class_map = {0:'glioma', 1:'meningioma', 2:'notumor', 3:'pituitary'}
    
    def predict_interface(input_image, auto_thresh, thresh_val, min_solidity, show_err_map,
                          p_name, p_id, p_age, p_gender, p_notes):
        if input_image is None:
            return None, {}, None, None, None, "### 📐 Estimated Tumor Dimensions:\n*No input image.*", None
            
        orig_pil = Image.fromarray(input_image).convert('L')
        
        # Classifier Prediction (built-in XPU/CUDA autocast, no IPEX)
        classify_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(advanced_preprocess),
            transforms.ToTensor(),
            transforms.Lambda(to_float_tensor),
            norm_transform
        ])(orig_pil).unsqueeze(0).to(device_classifier)
        if getattr(device_classifier, 'type', None) in ('xpu', 'cuda'):
            try:
                classify_tensor = classify_tensor.to(memory_format=torch.channels_last)
            except Exception:
                pass

        try:
            from train import _autocast_ctx as _app_autocast
            _app_ctx = _app_autocast(device_classifier, enabled=getattr(device_classifier, 'type', None) in ('xpu', 'cuda'))
        except Exception:
            from contextlib import nullcontext as _app_nc
            _app_ctx = _app_nc()

        with torch.no_grad(), _app_ctx:
            pred = classifier(classify_tensor)
            probs = torch.softmax(pred.float(), 1).squeeze(0)
            conf, idx = torch.max(probs, 0)
            pred_name = class_map[idx.item()]
            conf = conf.item()
            
        # Format label and confidence
        label_text = f"## Diagnosis: **{pred_name.upper()}** ({conf:.2%})"
        
        # Prepare probability dict for bar chart
        prob_dict = {class_map[i].capitalize(): float(probs[i].item()) for i in range(4)}
        
        # Grad-CAM Heatmap
        gradcam_pil = None
        classifier.eval()
        with torch.enable_grad():
            gradcam_tensor = classify_tensor.clone().detach()
            target_layer = classifier.densenet121.features.norm5
            with GradCAM(classifier, target_layer) as cam:
                try:
                    heatmap, _ = cam.generate_heatmap(gradcam_tensor, class_idx=idx.item())
                    gradcam_pil = get_gradcam_overlay(orig_pil, heatmap)
                except Exception as e:
                    print(f"Error generating Grad-CAM in WebApp: {e}")
                
        # Anomaly Detection & Outline
        outlined_pil = orig_pil.resize(image_size)
        err_map_pil = None
        metrics = {
            'total_area_px': 0.0,
            'total_area_cm2': 0.0,
            'width_px': 0.0,
            'width_cm': 0.0,
            'height_px': 0.0,
            'height_cm': 0.0,
            'has_tumor': False
        }
        
        if pred_name != 'notumor' and autoencoder is not None:
            ae_tensor = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Lambda(advanced_preprocess),
                transforms.ToTensor(),
                transforms.Lambda(to_float_tensor)
            ])(orig_pil).unsqueeze(0).to(device_outlining)
            if getattr(device_outlining, 'type', None) in ('xpu', 'cuda'):
                try:
                    ae_tensor = ae_tensor.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            try:
                from train import _autocast_ctx as _app_ae_autocast
                _app_ae_ctx = _app_ae_autocast(device_outlining, enabled=getattr(device_outlining, 'type', None) in ('xpu', 'cuda'))
            except Exception:
                from contextlib import nullcontext as _app_ae_nc
                _app_ae_ctx = _app_ae_nc()
            with torch.no_grad(), _app_ae_ctx:
                recon, att = autoencoder(ae_tensor)
                
            in_np = ae_tensor.squeeze(0).squeeze(0).float().cpu().numpy()
            recon_np = recon.squeeze(0).squeeze(0).float().cpu().numpy()
            att_np = att.squeeze(0).squeeze(0).float().cpu().numpy()
            
            # Reconstruction error map
            rec_err = np.abs(in_np - recon_np)
            err_map_resized = (rec_err - rec_err.min()) / (rec_err.max() - rec_err.min() + 1e-8)
            err_map_pil = Image.fromarray((err_map_resized * 255).astype(np.uint8)).convert('L')
            
            err_thresh = None if auto_thresh else thresh_val
            
            # Perform outline
            att_resized = F.interpolate(torch.from_numpy(att_np).unsqueeze(0).unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            mask, combined_map, contours = enhance_outline_detection(rec_err, att_resized, err_thresh, min_solidity)
            
            outlined_img = cv2.cvtColor(np.array(orig_pil.resize(image_size)), cv2.COLOR_GRAY2BGR)
            if contours:
                overlay = outlined_img.copy()
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.3, outlined_img, 0.7, 0, outlined_img)
                cv2.drawContours(outlined_img, contours, -1, (255, 0, 0), 2)
                
            outlined_pil = Image.fromarray(cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB))
            
            # Get metrics
            _, _, metrics = outline_tumor_enhanced(orig_pil, ae_tensor.squeeze(0), recon.squeeze(0), att.squeeze(0), image_size, err_thresh, min_solidity, show_plots=False)
        else:
            outlined_pil = orig_pil.resize(image_size)
            err_map_pil = Image.fromarray(np.zeros(image_size, dtype=np.uint8))
            
        # Format metrics markdown
        if metrics.get('has_tumor'):
            metrics_md = f"""### 📐 Estimated Tumor Dimensions:
* **Est. Area**: {metrics['total_area_cm2']:.2f} cm²
* **Max Width**: {metrics['width_cm']:.2f} cm
* **Max Height**: {metrics['height_cm']:.2f} cm"""
        else:
            metrics_md = "### 📐 Estimated Tumor Dimensions:\n* No tumor boundary detected for metric estimation."

        # Export Report (unique filename per request avoids concurrent-user overwrite)
        os.makedirs("classification_results", exist_ok=True)
        report_output_path = os.path.join("classification_results", f"diagnostic_report_{uuid.uuid4().hex[:8]}.html")
        generate_diagnostic_report(
            "uploaded_scan.png", pred_name, conf, metrics,
            orig_pil.resize(image_size), outlined_pil, gradcam_pil,
            report_output_path, str(device_classifier),
            patient_name=p_name, patient_id=p_id, age=p_age, gender=p_gender, physician_notes=p_notes
        )

        # Log to SQLite patient database
        log_diagnostic_run("uploaded_scan.png", pred_name, conf, metrics, report_output_path,
                           patient_name=p_name, patient_id=p_id, age=p_age, gender=p_gender, physician_notes=p_notes)

        outputs = [label_text, prob_dict, outlined_pil, gradcam_pil]
        if show_err_map:
            outputs.append(err_map_pil)
        else:
            outputs.append(None)
        
        outputs.append(metrics_md)
        outputs.append(report_output_path)
        outputs.append(load_records_table())
        outputs.append(get_stats_html())
        return tuple(outputs)

    # Database loading functions for UI
    def load_records_table(search_query=None):
        df = fetch_diagnostic_records(search_query)
        if df.empty:
            return pd.DataFrame(columns=['ID', 'Timestamp', 'Filename', 'Patient Name', 'Patient ID', 'Age', 'Gender', 'Diagnosis', 'Confidence', 'Area (cm²)', 'Width (cm)', 'Height (cm)', 'Notes', 'Report Path'])
        # Rename columns for presentation
        df.columns = ['ID', 'Timestamp', 'Filename', 'Patient Name', 'Patient ID', 'Age', 'Gender', 'Diagnosis', 'Confidence', 'Area (cm²)', 'Width (cm)', 'Height (cm)', 'Notes', 'Report Path']
        # Format columns
        df['Confidence'] = df['Confidence'].map(lambda x: f"{x:.2%}")
        df['Area (cm²)'] = df['Area (cm²)'].map(lambda x: f"{x:.2f}" if x > 0 else "0.00")
        df['Width (cm)'] = df['Width (cm)'].map(lambda x: f"{x:.2f}" if x > 0 else "0.00")
        df['Height (cm)'] = df['Height (cm)'].map(lambda x: f"{x:.2f}" if x > 0 else "0.00")
        df['Age'] = df['Age'].map(lambda x: str(int(x)) if not pd.isna(x) and x > 0 else "N/A")
        df['Patient Name'] = df['Patient Name'].fillna("Unknown")
        df['Patient ID'] = df['Patient ID'].fillna("N/A")
        df['Gender'] = df['Gender'].fillna("N/A")
        df['Notes'] = df['Notes'].fillna("")
        return df

    def get_stats_html():
        stats = get_database_stats()
        dist_items = "".join([f"<li style='margin-bottom: 5px;'><strong>{k.upper()}</strong>: {v} scans</li>" for k, v in stats['class_dist'].items()])
        if not dist_items:
            dist_items = "<li>No records logged yet.</li>"
            
        html = f"""
        <div style="background-color: #F8F9F9; padding: 20px; border-radius: 8px; border: 1px solid #E1E8ED; margin-bottom: 20px; font-family: sans-serif;">
            <h3 style="margin-top:0; color: #2C3E50; border-bottom: 2px solid #E1E8ED; padding-bottom: 10px;">📊 Central Audit Statistics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #E1E8ED; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold; color: #2980B9;">{stats['total_scans']}</div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #7F8C8D; margin-top: 3px;">Total Scans Logged</div>
                </div>
                <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #E1E8ED; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold; color: #27AE60;">{stats['avg_area']:.2f} cm²</div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #7F8C8D; margin-top: 3px;">Avg Tumor Area</div>
                </div>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #34495E;">Diagnosis Distributions:</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #5D6D7E; line-height: 1.4;">
                {dist_items}
            </ul>
        </div>
        """
        return html

    custom_css = """
    .container { max-width: 1200px; margin: auto; padding: 20px; }
    .title { text-align: center; font-family: 'Outfit', sans-serif; color: #2C3E50; margin-bottom: 30px; }
    .diagnosis-card { background-color: #F8F9F9; padding: 15px; border-radius: 8px; border-left: 5px solid #2980b9; margin-bottom: 20px; }
    .image-preview { border: 2px dashed #BDC3C7; border-radius: 8px; }
    """
    
    with gr.Blocks(css=custom_css, title="Brain Tumor AI Diagnostic Suite") as demo:
        gr.Markdown(
            """
            # 🧠 Brain Tumor AI Diagnostic & Localization Suite
            *Leveraging custom DenseNet121-CBAM classification and anomaly-based Autoencoder tumor localization.*
            """
        )
        
        with gr.Tabs():
            with gr.Tab("🧠 New Diagnosis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Input MRI Scan")
                        input_image = gr.Image(type="numpy", label="Upload Grayscale MRI Image", elem_classes="image-preview")
                        
                        with gr.Accordion("📋 Patient Information (Optional)", open=False):
                            p_name = gr.Textbox(label="Patient Name", placeholder="e.g. John Doe")
                            p_id = gr.Textbox(label="Patient ID / Record #", placeholder="e.g. MRN-55102")
                            p_age = gr.Number(label="Age", minimum=0, maximum=120, precision=0, value=None)
                            p_gender = gr.Dropdown(choices=["", "Male", "Female", "Other"], label="Gender", value="")
                            p_notes = gr.Textbox(label="Physician Notes / Observations", placeholder="Enter clinical comments...", lines=3)

                        with gr.Accordion("⚙️ Localization Parameters", open=True):
                            auto_thresh = gr.Checkbox(value=True, label="Auto Threshold (Otsu)")
                            thresh_val = gr.Slider(minimum=10, maximum=240, value=120, step=5, label="Manual Threshold")
                            min_solidity = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="Min Solidity Filter")
                            show_err_map = gr.Checkbox(value=True, label="Show Reconstruction Error Map")
                        
                        submit_btn = gr.Button("⚡ Analyze MRI Scan", variant="primary")
                        
                    with gr.Column(scale=2):
                        diagnosis_out = gr.Markdown("### Click 'Analyze MRI Scan' to see results.", elem_classes="diagnosis-card")
                        
                        with gr.Row():
                            prob_chart = gr.Label(num_top_classes=4, label="Class Probabilities")
                            with gr.Column():
                                metrics_out = gr.Markdown("### 📐 Estimated Tumor Dimensions:\n*Run scan to see metrics.*")
                                report_out = gr.File(label="📥 Printable Patient Report", visible=True)
                            
                        with gr.Row():
                            with gr.Tab("🎯 Tumor Localization"):
                                outlined_out = gr.Image(type="pil", label="Outlined Tumor Anomaly Overlay")
                            with gr.Tab("💡 Grad-CAM (XAI)"):
                                gradcam_out = gr.Image(type="pil", label="Classifier Attention Map")
                            with gr.Tab("⚠️ Reconstruction Error"):
                                err_map_out = gr.Image(type="pil", label="AE Reconstruction Error Map")
                                
            with gr.Tab("📋 Patient Record History"):
                gr.Markdown("### 🔍 AI Diagnosis Audit Logs")
                with gr.Row():
                    with gr.Column(scale=1):
                        stats_panel = gr.HTML(value=get_stats_html())
                        search_input = gr.Textbox(placeholder="Search by name, ID, or diagnosis...", label="🔍 Search Records")
                        search_btn = gr.Button("Search Records", variant="primary")
                        refresh_btn = gr.Button("🔄 Refresh Database", variant="secondary")
                        
                        gr.Markdown("---")
                        export_btn = gr.Button("📥 Export Full DB to CSV", variant="secondary")
                        export_file = gr.File(label="CSV Export Download", visible=True)
                        
                        gr.Markdown("---")
                        with gr.Accordion("🗑️ Delete Diagnostic Log", open=False):
                            delete_id = gr.Number(label="Record ID to Delete", precision=0)
                            delete_btn = gr.Button("Delete Log Entry", variant="stop")
                            delete_status = gr.Markdown("")
                            
                    with gr.Column(scale=3):
                        records_df = gr.Dataframe(value=load_records_table(), label="Audit Log Explorer", interactive=False)
                        
                def search_records(query):
                    return load_records_table(query), get_stats_html()
                    
                def refresh_records():
                    return load_records_table(), get_stats_html()
                    
                def delete_record_handler(rec_id):
                    if rec_id is None:
                        return "Please enter a valid Record ID.", load_records_table(), get_stats_html()
                    success = delete_diagnostic_record(int(rec_id))
                    if success:
                        return f"Successfully deleted record #{int(rec_id)}.", load_records_table(), get_stats_html()
                    else:
                        return f"Failed to delete record #{int(rec_id)}.", load_records_table(), get_stats_html()
                    
                submit_btn.click(
                    fn=predict_interface,
                    inputs=[input_image, auto_thresh, thresh_val, min_solidity, show_err_map,
                            p_name, p_id, p_age, p_gender, p_notes],
                    outputs=[diagnosis_out, prob_chart, outlined_out, gradcam_out, err_map_out, metrics_out, report_out, records_df, stats_panel]
                )

                search_btn.click(fn=search_records, inputs=[search_input], outputs=[records_df, stats_panel])
                search_input.submit(fn=search_records, inputs=[search_input], outputs=[records_df, stats_panel])
                refresh_btn.click(fn=refresh_records, outputs=[records_df, stats_panel])
                export_btn.click(fn=export_database_csv, outputs=[export_file])
                delete_btn.click(fn=delete_record_handler, inputs=[delete_id], outputs=[delete_status, records_df, stats_panel])
        
    print("\nStarting Gradio App server...")
    demo.launch(share=False)
