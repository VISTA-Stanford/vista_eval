import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

def pad_to_512(img):
    """
    Resizes image if > 512, then pads to 512x512 with right/bottom padding.
    """
    w, h = img.size
    size = 512

    # --- New Logic: Resize if larger than 512 ---
    if w > size or h > size:
        # Determine the scaling factor to keep the largest side at 512
        scale = size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize using high-quality resampling
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Update dimensions for the padding step
        w, h = img.size

    # --- Original Logic: Pad to 512 ---
    # (If the image is already 512x512, this will result in 0 padding)
    pad_w = max(0, size - w)
    pad_h = max(0, size - h)
    
    # Optimization: return immediately if no padding is needed
    if pad_w == 0 and pad_h == 0:
        return img
        
    return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)

def normalize_slice(slice_data):
    """
    Normalize slice data to 0-255 range for visualization.
    Following the logic from vqa_dataset.py
    """
    # Handle potential NaN or inf values
    if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
        slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to 0-255
    if slice_data.max() > slice_data.min():
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    else:
        slice_data = np.zeros_like(slice_data, dtype=np.uint8)
    return slice_data