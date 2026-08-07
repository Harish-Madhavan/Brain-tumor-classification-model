import os
import datetime
import random
import io
import base64

def generate_diagnostic_report(img_path, pred_name, conf, metrics, original_pil, outlined_pil, gradcam_pil, output_path, device_name,
                               patient_name=None, patient_id=None, age=None, gender=None, physician_notes=None):
    def pil_to_base64(pil_img):
        if pil_img is None:
            return ""
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"{random.randint(1000, 9999)}"
    filename = os.path.basename(img_path)
    
    original_b64 = pil_to_base64(original_pil)
    outlined_b64 = pil_to_base64(outlined_pil)
    gradcam_b64 = pil_to_base64(gradcam_pil)
    
    # Format metrics
    if metrics and metrics.get('has_tumor'):
        tumor_area = f"{metrics['total_area_cm2']:.2f} cm²"
        tumor_width = f"{metrics['width_cm']:.2f} cm"
        tumor_height = f"{metrics['height_cm']:.2f} cm"
    else:
        tumor_area = "N/A (No Tumor)"
        tumor_width = "N/A"
        tumor_height = "N/A"
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Brain Tumor AI Diagnostic Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; margin: 0; padding: 40px; background-color: #f9f9f9; }}
        .report-card {{ max-width: 900px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-top: 8px solid #2980b9; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; color: #2c3e50; font-size: 26px; }}
        .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .meta-table td {{ padding: 12px 15px; border: 1px solid #e1e8ed; font-size: 14px; }}
        .meta-table td.label {{ font-weight: bold; background-color: #f8f9fa; color: #2c3e50; width: 25%; }}
        .diagnosis-box {{ background-color: #ebf5fb; border-left: 6px solid #2980b9; padding: 20px; border-radius: 4px; margin-bottom: 35px; }}
        .diagnosis-box h2 {{ margin: 0 0 10px 0; color: #2980b9; font-size: 20px; }}
        .diagnosis-box p {{ margin: 0; font-size: 16px; font-weight: 500; }}
        .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 35px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; border: 1px solid #e1e8ed; }}
        .metric-card .value {{ font-size: 22px; font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .metric-card .title {{ font-size: 12px; text-transform: uppercase; color: #7f8c8d; font-weight: 600; }}
        .images-container {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
        .image-box {{ border: 1px solid #e1e8ed; border-radius: 6px; padding: 10px; text-align: center; background: #fff; }}
        .image-box img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto 10px auto; }}
        .image-box .caption {{ font-size: 12px; font-weight: 600; color: #7f8c8d; }}
        .footer {{ border-top: 2px solid #eee; padding-top: 20px; margin-top: 40px; font-size: 12px; color: #95a5a6; display: flex; justify-content: space-between; }}
        .signatures {{ display: flex; justify-content: space-between; margin-top: 50px; }}
        .signature-line {{ border-top: 1px dashed #7f8c8d; width: 200px; text-align: center; padding-top: 5px; font-size: 13px; color: #7f8c8d; }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .report-card {{ box-shadow: none; padding: 0; border: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <h1>Brain Tumor AI Diagnostic Report</h1>
                <span style="color: #7f8c8d; font-size: 13px;">Clinical Decision Support Output</span>
            </div>
            <div style="text-align: right;">
                <span style="font-weight: bold; color: #2c3e50;">ID: BTR-{report_id}</span><br>
                <span style="font-size: 13px; color: #7f8c8d;">Generated: {timestamp}</span>
            </div>
        </div>

        <table class="meta-table">
            <tr>
                <td class="label">Patient Name</td>
                <td>{patient_name if patient_name else "Anonymous / Unknown"}</td>
                <td class="label">Patient ID</td>
                <td>{patient_id if patient_id else "N/A"}</td>
            </tr>
            <tr>
                <td class="label">Age</td>
                <td>{f"{age} years old" if age else "N/A"}</td>
                <td class="label">Gender</td>
                <td>{gender if gender else "N/A"}</td>
            </tr>
            <tr>
                <td class="label">Patient Scan File</td>
                <td>{filename}</td>
                <td class="label">System Version</td>
                <td>v1.2.0 (DenseNet121-CBAM)</td>
            </tr>
            <tr>
                <td class="label">Hardware Device</td>
                <td>{device_name}</td>
                <td class="label">Localization Mode</td>
                <td>Dual (SSIM AE + Grad-CAM)</td>
            </tr>
        </table>

        <div class="diagnosis-box">
            <h2>Primary Diagnosis Classification</h2>
            <p>Detected Class: <span style="color: #c0392b; font-weight: bold; font-size: 18px;">{pred_name.upper()}</span> with <span style="font-weight: bold;">{conf:.2%}</span> classification confidence.</p>
        </div>

        <h3 style="color: #2c3e50; border-bottom: 2px solid #f2f2f2; padding-bottom: 8px; margin-bottom: 20px;">Estimated Tumor Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{tumor_area}</div>
                <div class="title">Est. Area (cm²)</div>
            </div>
            <div class="metric-card">
                <div class="value">{tumor_width}</div>
                <div class="title">Est. Max Width (cm)</div>
            </div>
            <div class="metric-card">
                <div class="value">{tumor_height}</div>
                <div class="title">Est. Max Height (cm)</div>
            </div>
        </div>

        {f"""<div style="background-color: #fdfefe; border-left: 6px solid #7f8c8d; padding: 15px 20px; border-radius: 4px; margin-bottom: 35px; border: 1px solid #e1e8ed; border-left-width: 6px;">
            <h4 style="margin: 0 0 8px 0; color: #34495e; font-size: 15px;">Physician Observations / Radiologist Notes:</h4>
            <p style="margin: 0; font-size: 14px; color: #5d6d7e; line-height: 1.4; white-space: pre-wrap;">{physician_notes}</p>
        </div>""" if physician_notes else ""}

        <h3 style="color: #2c3e50; border-bottom: 2px solid #f2f2f2; padding-bottom: 8px; margin-bottom: 20px;">Visual Interpretability Panels</h3>
        <div class="images-container">
            <div class="image-box">
                <img src="{original_b64}" alt="Original MRI">
                <div class="caption">Original MRI</div>
            </div>
            <div class="image-box">
                <img src="{outlined_b64}" alt="Localization Outline">
                <div class="caption">Tumor Localization Outline</div>
            </div>
            <div class="image-box">
                <img src="{gradcam_b64}" alt="Grad-CAM Heatmap">
                <div class="caption">Grad-CAM Attention Map</div>
            </div>
        </div>

        <div class="signatures">
            <div>
                <br><br>
                <div class="signature-line">Radiologist Signature</div>
            </div>
            <div style="text-align: right;">
                <br><br>
                <div class="signature-line">Review Date</div>
            </div>
        </div>

        <div class="footer">
            <span>Powered by PyTorch & Gradio Medical AI Pipeline</span>
            <span>Confidential - Medical Use Only</span>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Diagnostic report saved successfully to {output_path}")
