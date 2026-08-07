import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activations(module, input, output):
            self.activations = output
            
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.hooks.append(self.target_layer.register_full_backward_hook(save_gradients))
        else:
            self.hooks.append(self.target_layer.register_backward_hook(save_gradients))
        self.hooks.append(self.target_layer.register_forward_hook(save_activations))
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def get_gradcam_overlay(orig_pil, heatmap, alpha=0.4):
    orig_np = np.array(orig_pil)
    if len(orig_np.shape) == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    
    # Scale heatmap to [0, 255] and apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend images
    overlay_np = cv2.addWeighted(heatmap_colored, alpha, orig_np, 1.0 - alpha, 0)
    return Image.fromarray(overlay_np)
