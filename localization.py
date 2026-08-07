import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def enhance_outline_detection(rec_err, att_map, threshold=None, min_solidity=0.6):
    rec_norm = (rec_err - rec_err.min()) / (rec_err.max() - rec_err.min() + 1e-8)
    att_norm = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    combined = cv2.medianBlur(
        cv2.GaussianBlur((np.clip(0.4 * rec_norm + 0.6 * att_norm, 0, 1) * 255).astype(np.uint8), (5, 5), 0), 5
    )
    if threshold is None:
        if combined.max() > 0:
            threshold, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            threshold = 255.0
            
    mask = cv2.morphologyEx(
        cv2.morphologyEx((combined > threshold).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
        cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if 50 < area < (mask.shape[0] * mask.shape[1] * 0.6) and hull_area > 0 and (area / hull_area) >= min_solidity:
            valid_contours.append(c)
    return mask, (combined.astype(np.float32) / 255.0), valid_contours

def outline_tumor_enhanced(orig_pil, in_tensor, recon_tensor, att_tensor, size, err_thresh, solidity, show_plots):
    in_np = in_tensor.squeeze().cpu().numpy()
    recon_np = recon_tensor.squeeze().cpu().numpy()
    att_np = att_tensor.squeeze().cpu().numpy()
    
    orig_resized_np = np.array(orig_pil.resize(size))
    rec_err = np.abs(in_np - recon_np)
    att_resized = F.interpolate(
        torch.from_numpy(att_np).unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    mask, combined_map, contours = enhance_outline_detection(rec_err, att_resized, err_thresh, solidity)
    outlined_img = cv2.cvtColor(orig_resized_np, cv2.COLOR_GRAY2BGR)
    if contours:
        overlay = outlined_img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, outlined_img, 0.7, 0, outlined_img)
        cv2.drawContours(outlined_img, contours, -1, (255, 0, 0), 2)
        
    pixel_spacing_mm = 1.15
    pixel_to_cm = pixel_spacing_mm / 10.0
    
    total_area_px = 0.0
    width_px, height_px = 0.0, 0.0
    width_cm, height_cm = 0.0, 0.0
    total_area_cm2 = 0.0
    
    if contours:
        x_min, y_min, x_max, y_max = size[0], size[1], 0, 0
        for c in contours:
            total_area_px += cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            
        width_px = x_max - x_min
        height_px = y_max - y_min
        width_cm = width_px * pixel_to_cm
        height_cm = height_px * pixel_to_cm
        total_area_cm2 = total_area_px * (pixel_to_cm ** 2)
        
    metrics = {
        'total_area_px': total_area_px,
        'total_area_cm2': total_area_cm2,
        'width_px': width_px,
        'width_cm': width_cm,
        'height_px': height_px,
        'height_cm': height_cm,
        'has_tumor': len(contours) > 0
    }
    
    if show_plots:
        fig, axes = plt.subplots(1, 8, figsize=(20, 5))
        titles = ['Original', 'AE Input', 'AE Recon', 'Error', 'Attention', 'Combined', 'Mask', 'Result']
        imgs = [orig_resized_np, in_np, recon_np, rec_err, att_resized, combined_map, mask, cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB)]
        cmaps = ['gray', 'gray', 'gray', 'hot', 'viridis', 'hot', 'gray', None]
        for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    return Image.fromarray(orig_resized_np), Image.fromarray(cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB)), metrics
