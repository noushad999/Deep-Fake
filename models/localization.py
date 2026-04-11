"""
GradCAM++ Localization Module
Generates forgery heatmaps showing manipulated regions.
"""
import torch
import numpy as np
import cv2
from typing import Tuple, Optional
from pathlib import Path


class GradCAMLocalization:
    """
    GradCAM++ implementation for deepfake localization.
    Produces pixel-level forgery probability heatmaps.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        """
        Args:
            model: The full model or backbone to compute GradCAM on.
            target_layer: Layer to extract gradients from. If None, uses last conv layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        if self.target_layer is None:
            # Default: use spatial stream's last conv layer
            self.target_layer = self.model.spatial_stream.backbone.conv_head
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _compute_gradcampp(self) -> np.ndarray:
        """
        Compute GradCAM++ heatmap from stored activations and gradients.
        
        Returns:
            Heatmap (H, W) normalized to [0, 1]
        """
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Get dimensions
        channels, h, w = gradients.shape
        
        # GradCAM++ weighting
        gradients = gradients.reshape(channels, -1)
        activations = activations.reshape(channels, -1)
        
        # Compute alpha coefficients
        alpha_num = gradients ** 2
        alpha_den = 2 * gradients ** 2 + \
                    sum(activations * gradients ** 3 for activations in [activations])
        alpha_den = np.where(alpha_den != 0.0, alpha_den, 1.0)
        alpha = alpha_num / (alpha_den + 1e-8)
        
        # Compute weights
        weights = np.maximum(gradients, 0)
        weights = weights * alpha
        weights = weights.sum(axis=1)  # Sum over spatial dimensions
        
        # Compute GradCAM++
        cam = np.zeros((h, w), dtype=np.float32)
        for c in range(channels):
            cam += weights[c] * activations[c].reshape(h, w)
        
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap for an input.
        
        Args:
            input_tensor: [1, 3, H, W] normalized input
            target_class: Target class (0=real, 1=fake). If None, uses predicted class.
        
        Returns:
            Heatmap [H, W] in range [0, 1]
        """
        self.model.eval()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        output, _ = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = int(torch.sigmoid(output) > 0.5)
        
        # Backward pass on target
        target_output = output[0, target_class] if output.shape[1] > 1 else output[0, 0]
        target_output.backward()
        
        # Compute GradCAM++
        heatmap = self._compute_gradcampp()
        
        return heatmap
    
    def visualize_heatmap(
        self,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            input_tensor: [1, 3, H, W] normalized input
            heatmap: [H, W] heatmap in [0, 1]
            colormap: OpenCV colormap
        
        Returns:
            Visualization [H, W, 3] in RGB
        """
        # Denormalize input
        img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Unnormalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Resize heatmap to match image
        if heatmap.shape != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Blend
        alpha = 0.5
        vis = (1 - alpha) * img + alpha * heatmap_colored
        vis = np.clip(vis, 0, 1)
        
        return np.uint8(255 * vis)
    
    def save_heatmap(
        self,
        input_tensor: torch.Tensor,
        output_path: str,
        target_class: Optional[int] = None,
        save_visualization: bool = True
    ) -> str:
        """
        Generate and save heatmap to disk.
        
        Args:
            input_tensor: [1, 3, H, W] input
            output_path: Path to save
            target_class: Target class
            save_visualization: If True, save overlay; else save raw heatmap
        
        Returns:
            Path to saved file
        """
        heatmap = self.generate_heatmap(input_tensor, target_class)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_visualization:
            vis = self.visualize_heatmap(input_tensor, heatmap)
            cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        else:
            # Save raw heatmap as image
            heatmap_uint8 = np.uint8(255 * heatmap)
            cv2.imwrite(str(output_path), heatmap_uint8)
        
        return str(output_path)


def generate_batch_heatmaps(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    device: torch.device
) -> list:
    """
    Generate heatmaps for a batch of images.
    
    Args:
        model: Trained model
        images: [B, 3, H, W] batch
        labels: [B] ground truth labels
        output_dir: Directory to save heatmaps
        device: Device to run on
    
    Returns:
        List of saved heatmap paths
    """
    model.eval()
    cam = GradCAMLocalization(model)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    with torch.no_grad():
        for i in range(images.shape[0]):
            img = images[i:i+1].to(device)
            label = int(labels[i].item())
            
            # Generate heatmap
            heatmap = cam.generate_heatmap(img, target_class=label)
            vis = cam.visualize_heatmap(img, heatmap)
            
            # Save
            pred = int((torch.sigmoid(model(img)[0]) > 0.5).item())
            filename = f"sample_{i}_true{label}_pred{pred}.jpg"
            filepath = output_dir / filename
            
            cv2.imwrite(str(filepath), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(filepath))
    
    cam.remove_hooks()
    return saved_paths


if __name__ == "__main__":
    print("="*60)
    print("TESTING: GradCAM++ Localization Module")
    print("="*60)
    print("\nGradCAM++ module loaded successfully.")
    print("Note: Full test requires a trained model.")
    print("Run evaluate.py to test heatmap generation.")
