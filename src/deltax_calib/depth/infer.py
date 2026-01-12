"""
Monocular Depth Inference Module

Supports multiple depth estimation backends:
- depth_anything: Depth Anything V2 (recommended - best quality)
- midas: Intel MiDaS (older, more compressed depth range)
- stub: Constant depth for testing

Note: These models output relative disparity, not metric depth. We use LiDAR
points to fit a scale factor and convert to metric depth in the fusion step.

Author: DeltaX Assignment Submission
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class DepthInference:
    def __init__(self, method: str = "stub", model_path: Optional[str] = None, device: str = "cpu"):
        self.method = method
        self.model_path = model_path
        self.device = device
        self.model = None
        self.transform = None
        self.processor = None
        if method == "midas":
            self._load_midas()
        elif method == "depth_anything":
            self._load_depth_anything()
        elif method == "torchscript":
            if not model_path:
                raise ValueError("model_path is required for torchscript depth.")
            self.model = torch.jit.load(model_path, map_location=device)
            self.model.eval()

    def _load_midas(self) -> None:
        model_type = "DPT_Hybrid"
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.dpt_transform

    def _load_depth_anything(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        # Use Depth Anything V2 base model
        model_name = "depth-anything/Depth-Anything-V2-Base-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Infer depth from image. Returns relative disparity (needs scaling)."""
        height, width = image.shape[:2]
        if self.method == "stub":
            return np.full((height, width), 20.0, dtype=np.float32)

        if self.method == "torchscript":
            img = image.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(tensor).squeeze()
            if pred.ndim == 2:
                pred = pred.unsqueeze(0)
            pred = F.interpolate(pred.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
            return pred.squeeze().cpu().numpy().astype(np.float32)

        if self.method == "midas":
            img = image.astype(np.float32) / 255.0
            input_batch = self.transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            # Return disparity directly - will be converted to depth with LiDAR scaling
            disparity = prediction.cpu().numpy().astype(np.float32)
            return disparity

        if self.method == "depth_anything":
            from PIL import Image
            pil_image = Image.fromarray(image)
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            # Interpolate to original size
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            # Depth Anything outputs disparity-like values (higher = closer)
            disparity = prediction.cpu().numpy().astype(np.float32)
            return disparity

        raise ValueError(f"Unsupported depth method: {self.method}")
