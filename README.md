# Brain Tumor Classification & Localization Model

A deep learning diagnostic assistant utilizing a custom **DenseNet121-CBAM** attention model for 4-class classification (Glioma, Meningioma, No Tumor, Pituitary) and an anomaly-detection-based **U-Net Autoencoder** for pixel-level tumor localization.

---

## 🚀 Key Features

1. **Gradio Web Application Interface**: An interactive browser-based dashboard to upload MRI scans, adjust localization parameters (solidity filters, thresholding) in real-time, view classification probability charts, and visualize Grad-CAM activation overlays. Includes a collapsible Patient Information form panel (Name, ID, Age, Gender, Notes).
2. **Local SQLite Patient Database**: A lightweight database (`patient_records.db`) storing a central audit trail of all AI-assisted diagnoses, scan filenames, timestamps, prediction classes, confidences, tumor dimensions, patient demographics, and physician comments.
3. **Interactive Patient Record History**: A searchable dashboard tab in the Gradio app showing summary system statistics, a responsive data explorer table for audit logs, administrative deletion tools, and a full database exporter generating downloadable CSV files.
4. **Clinical Tumor Metric Estimation**: Automatically calculates pixel-level and physical tumor area (cm²), max width, and max height based on standard pixel spacing (1.15 mm/pixel).
5. **Printable Diagnostic Report Generator**: Exports a professional, printable HTML diagnostic report containing patient demographics, clinical radiologist notes, classification confidence, tumor metrics, and side-by-side visualization cards.
6. **Explainable AI (Grad-CAM)**: Visualizes the attention maps of the DenseNet feature extractor to explain the model's diagnostic reasoning.
7. **Batch Inference Pipeline**: Processes directories of MRI images, outputs contoured tumor outlines, Grad-CAM attention maps, and compiles results including tumor dimensions into a clean CSV report.
8. **Unified Argparse CLI**: Fully controllable pipeline supporting Training, Evaluation, Single Prediction, and Web Interface modes.

---

## 🛠️ Usage Guide

First, install the required packages (including `gradio` if running the web application):
```bash
pip install torch torchvision numpy scikit-learn opencv-python matplotlib tqdm scikit-image pytorch-msssim seaborn pandas openpyxl gradio
```

### 1. Launch the Gradio Web App
Start the interactive dashboard locally:
```bash
python torchm_denseNet.py --mode app
```
Once launched, open the local address shown in your terminal (typically `http://127.0.0.1:7860`). Run scans and click the **Patient Record History** tab to explore SQLite audit logs.

### 2. Predict on a Single MRI Scan (with Patient Metadata, PDF Report, and DB Logging)
Analyze a MRI scan, save tumor outline, Grad-CAM heatmap, export report with patient details, and log the run:
```bash
python torchm_denseNet.py --mode predict --image "path/to/mri.jpg" --gradcam --export_report --patient_name "Alice Smith" --patient_id "MRN-1122" --age 28 --gender "Female" --notes "Clinical follow-up on pituitary region."
```

### 3. Run Batch Inference (with CSV Export and DB Logging)
Process a folder of MRI scans, export a `batch_inference_report.csv`, and automatically log all runs to the SQLite database:
```bash
python torchm_denseNet.py --mode batch --input_dir "path/to/images" --output_dir "output_results" --gradcam
```

### 4. Evaluate on Test Set
Evaluate the classifier's performance and generate a confusion matrix, classification report, and misclassifications plot:
```bash
python torchm_denseNet.py --mode evaluate --data_dir "training1"
```

### 5. Train Models from Scratch
Train both the classifier and autoencoder models:
```bash
python torchm_denseNet.py --mode train --data_dir "training1"
```

---

## 📁 CLI Arguments Reference

| Argument | Choices / Default | Description |
|---|---|---|
| `--mode` | `app` (default), `predict`, `batch`, `train`, `evaluate` | Execution mode for the pipeline |
| `--data_dir` | Local folder `training1` | Path to dataset directory |
| `--results_dir`| `classification_results` | Output directory for reports, plots, and weights |
| `--c_path` | Resolved automatically | Path to classifier model weights (`.pth`) |
| `--ae_path` | Resolved automatically | Path to autoencoder model weights (`.pth`) |
| `--image` | None | Path to input image for `predict` mode |
| `--input_dir` | None | Input directory of images for `batch` mode |
| `--output_dir`| `batch_output` | Output directory for batch results |
| `--gradcam` | Flag (off by default) | Generate Grad-CAM activation maps in predict/batch modes |
| `--export_report`| Flag (off by default) | Export printable HTML diagnostic report in predict mode |

---

For the technical report, see [Project Report](Project_Report.md).
