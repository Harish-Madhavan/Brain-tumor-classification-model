# Project Report: Brain Tumor Classification and Localization

## 1. Abstract
* **Purpose**: Develop an automated, explainable clinical decision-support pipeline for the classification and localization of brain tumors in MRI scans.
* **Methodology**: Combines a deep-learning classification network (DenseNet121 augmented with a Convolutional Block Attention Module - CBAM) and an unsupervised localization network (U-Net-style Enhanced Autoencoder with skip connections and CBAM).
* **Classification Results**: Achieves an overall accuracy of 98.0% across 1,054 test images, classifying scans into Glioma, Meningioma, Pituitary, and No Tumor.
* **Localization Results**: Employs anomaly detection by training the autoencoder exclusively on healthy MRI scans, segmenting tumor regions using reconstruction error and bottleneck attention maps without manual pixel-level annotation.
* **Clinical Value**: Provides rapid, interpretable diagnosis (sub-0.15s inference time) to aid radiologists in clinical workflows and surgical planning.

## 2. Introduction
* **Clinical Context**: Brain tumors are life-threatening central nervous system malignancies requiring prompt and precise diagnosis.
* **Traditional Workflow**: Radiologists manually screen MRI slices, a process prone to subjectivity, diagnostic fatigue, and throughput bottlenecks.
* **Deep Learning Integration**: Computer-aided diagnosis (CAD) leveraging Convolutional Neural Networks (CNNs) accelerates scanning workflows and reduces diagnostic error rates.
* **Project Scope**: An end-to-end medical imaging pipeline integrating CLAHE/Gaussian preprocessing, multi-class classification, attention profiling, and unsupervised lesion localization.

## 3. Problem Statement
* **Manual Bottleneck**: Manual tumor typing and localization from multi-slice MRI scans is time-consuming and subjective.
* **Label Scarcity**: State-of-the-art segmentation networks (e.g., standard U-Nets) require thousands of manually drawn, expert-annotated pixel masks, which are extremely costly and scarce.
* **Attention Deficit in CNNs**: Conventional architectures lack mechanisms to selectively focus on tumor regions, often leading to false positives guided by background noise or skull tissue.
* **Dataset Imbalance**: Healthcare data routinely exhibits class imbalance, leading to skewed model training and high false negative rates for rarer tumor types.

## 4. Objectives
* **High-Accuracy Classification**: Design a multi-class model achieving $>95\%$ macro F1-score across Glioma, Meningioma, Pituitary, and No Tumor classes.
* **Unsupervised Delineation**: Implement an anomaly-detection-based system to localize tumors without requiring pixel-level annotation masks during training.
* **Explainable AI (XAI)**: Embed channel and spatial attention mechanisms (CBAM) to trace and explain classification and localization decisions.
* **Efficiency and Usability**: Deliver a lightweight execution script achieving sub-2-second inference latency on standard systems.

## 5. Existing Systems
* **Fully Supervised Segmenters**: Models like standard U-Net, SegNet, or DeepLab require pixel-perfect binary masks for training, which limits scalability.
* **Traditional CNN Classifiers**: Standard ResNet or VGG architectures classify images but lack interpretability, operating as "black boxes" that fail to justify predictions visually.
* **Binary Diagnostic Systems**: Many existing clinical classification systems only perform binary categorization (Tumor vs. No Tumor), failing to differentiate tumor subtypes.
* **High Resource Requirements**: Heavy architectures require substantial local compute resources (multi-GPU environments), rendering them impractical for localized clinical deployment.

## 6. Proposed System
* **Dual-Network Pipeline**: Integrates a custom DenseNet121-CBAM classifier and a U-Net style Enhanced Autoencoder for visual anomaly mapping.
* **Advanced Image Preprocessing**: Employs Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize contrast variations, followed by a Gaussian blur to filter high-frequency noise.
* **Attention-Guided Feature Extraction**: Inserts a Convolutional Block Attention Module (CBAM) to refine spatial and channel features, emphasizing tumor regions while suppressing background noise.
* **Anomaly-Based Outlining**: Trains the autoencoder exclusively on healthy MRI scans. When evaluating a scan with a tumor, the network fails to reconstruct the pathological regions, generating high reconstruction errors that reveal the exact tumor location.
* **Optimized Loss Functions**: Uses weighted Cross-Entropy loss to combat dataset imbalance, and a hybrid MS-SSIM + MSE loss with attention regularization for high-fidelity autoencoder reconstructions.

## 7. System Modules
* **Data Ingestion & Augmentation**:
  * Loads single-channel (grayscale) 256x256 MRI scans.
  * Implements dynamic train-time augmentation (random flips, rotations, affine transforms, color jitter, and grayscale conversions) to improve generalization.
* **Preprocessing Module**:
  * Applies CLAHE (clip limit 2.0, 8x8 tile size) for local contrast enhancement.
  * Applies Gaussian Blurring (3x3 kernel) to suppress sensory and capture noise.
* **Classification Module (DenseNet121-CBAM)**:
  * Uses a pre-trained DenseNet121 backbone modified for single-channel grayscale input.
  * Embeds a CBAM block (channel and spatial attention) at the output of the dense feature extractor.
  * Uses a multi-layer classifier head with dropout layers (0.4/0.3) to prevent overfitting.
* **Localization Module (Enhanced Autoencoder)**:
  * Employs a U-Net encoder-decoder structure with skip connections for fine-grained feature reuse.
  * Integrates CBAM blocks after every convolutional block in both the encoder and decoder stages.
  * Employs a bottleneck attention layer to produce an auxiliary visual attention map.
* **Anomaly Post-Processor**:
  * Computes the absolute reconstruction error map (residual map) between input and reconstructed images.
  * Blends the normalized error map (40%) and bottleneck attention map (60%).
  * Applies Otsu's thresholding and morphological operations (closing/opening) to isolate boundaries and extract tumor contours.
* **Logging and Verification**:
  * Records epoch-by-epoch training metrics directly to an Excel log (`training_log_per_epoch.xlsx`).
  * Generates and saves evaluation plots (Confusion Matrix, Misclassified Examples, Training Curves).

## 8. Tech Stack
* **Programming Language**: Python 3.8+
* **Deep Learning Framework**: PyTorch (v2.0+)
* **Computer Vision Libraries**: OpenCV (cv2), Pillow (PIL), Scikit-Image
* **Data Science & Math**: NumPy, SciPy, Scikit-Learn, Pandas
* **Visualization Tools**: Matplotlib, Seaborn
* **Progress Utilities**: tqdm
* **Accelerators**: NVIDIA CUDA and Intel XPU engines

## 9. Results & Achievements
* **Test Performance (1,054 Images)**:
  * **Overall Accuracy**: 98.0%
  * **Glioma**: Precision: 0.99 | Recall: 0.99 | F1-Score: 0.99 (Support: 243)
  * **Meningioma**: Precision: 0.98 | Recall: 0.96 | F1-Score: 0.97 (Support: 247)
  * **No Tumor**: Precision: 0.99 | Recall: 0.99 | F1-Score: 0.99 (Support: 300)
  * **Pituitary**: Precision: 0.97 | Recall: 0.99 | F1-Score: 0.98 (Support: 264)
* **Zero-Anomaly Calibration**: Successfully validates on healthy scans with zero false contours detected.
* **Unsupervised Delineation**: Delineates exact tumor outlines on positive scans using the error-attention fusion method, bypassing pixel-level target masks.
* **Inference Speed**: Accomplishes prediction and outlining in under 0.15 seconds per scan on GPU systems.

## 10. Future Enhancements
* **3D Volumetric Processing**: Extend models to handle full 3D MRI voxel volumes (MRI sequences) instead of isolated 2D slices.
* **Semi-Supervised Segmentation**: Incorporate a minimal set of manual pixel-level masks to transition the unsupervised localization into a highly precise semi-supervised segmenter.
* **Multi-Modal Support**: Train networks to fuse multi-modal MRI signals (T1, T2, FLAIR, Contrast-Enhanced T1) to improve diagnostic coverage.
* **Web Diagnostic Interface**: Build a web application using FastAPI and Streamlit/React for remote DICOM file upload and overlay visualization.
* **Federated Learning Framework**: Integrate decentralized training modules to learn from multi-institutional datasets without exposing patient health records (PHI).

## 11. Conclusion
* **Successful Pipeline Integration**: Developed a unified, modular model that achieves both high-accuracy classifier categorization and unsupervised visual localization.
* **Explainability Breakthrough**: Verified that CBAM attention modules guide the model to focus on tumor morphology, enhancing doctor trust in neural network outputs.
* **Cost-Effective Development**: Demonstrated that anomaly detection autoencoders bypass the need for manual, pixel-level training datasets while maintaining reliable tumor localization.
* **Translational Potential**: The high accuracy, fast runtimes, and clear visualizations demonstrate the feasibility of clinical translation as a diagnostic screening assistant.
