#!/usr/bin/env python3
"""
Create a PDF with all slices from a CT scan.
Takes the first task from config, finds a CT scan that exists,
and creates a PDF with all slices.
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image, ImageOps
from vista_run.utils.utils_inference import pad_to_512

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_valid_tasks(valid_tasks_path):
    """Load valid tasks JSON."""
    with open(valid_tasks_path, 'r') as f:
        return json.load(f)

def local_path_to_nifti_path(local_path_str):
    """
    Convert local_path to actual NIfTI file path.
    Following the logic from download_subsampled_ct.py
    """
    parts = local_path_str.split('/')
    filename_no_ext = parts[-1].replace('.zip', '')
    bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
    
    download_base = Path('/home/dcunhrya/downloaded_ct_scans')
    prefix = 'chaudhari_lab/ct_data/ct_scans/vista/nov25'
    nifti_path = download_base / prefix / bucket_filename
    
    return nifti_path

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

def create_ct_pdf():
    """Main function to create PDF with all CT slices."""
    
    # 1. Load config
    config_path = Path('/home/dcunhrya/vista_eval/configs/all_tasks.yaml')
    config = load_config(config_path)
    
    # 2. Get first task
    tasks = config.get('tasks', [])
    if not tasks:
        print("ERROR: No tasks found in config file")
        return
    
    first_task_name = tasks[0]
    print(f"Using first task: {first_task_name}")
    
    # 3. Load valid tasks to get task_source_csv
    base_path = Path(config['paths']['base_dir'])
    valid_tasks_path = base_path / config['paths']['valid_tasks']
    valid_tasks = load_valid_tasks(valid_tasks_path)
    
    # Find the task info
    task_info = None
    for task in valid_tasks:
        if task['task_name'] == first_task_name:
            task_info = task
            break
    
    if not task_info:
        print(f"ERROR: Task {first_task_name} not found in valid_tasks.json")
        return
    
    source_csv = task_info['task_source_csv']
    print(f"Task source CSV: {source_csv}")
    
    # 4. Load CSV file
    csv_path = base_path / source_csv / f"{first_task_name}.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 5. Find a CT scan that exists
    local_path_col = next((c for c in df.columns if c.lower() == 'local_path'), None)
    if not local_path_col:
        print("ERROR: No 'local_path' column found in CSV")
        return
    
    print(f"Found local_path column: {local_path_col}")
    print(f"Total rows: {len(df)}")
    
    # Find first row with existing CT scan
    nifti_path = None
    local_path_value = None
    
    for idx, row in df.iterrows():
        local_path_str = row[local_path_col]
        if pd.notna(local_path_str) and isinstance(local_path_str, str):
            nifti_path = local_path_to_nifti_path(local_path_str)
            if nifti_path.exists():
                local_path_value = local_path_str
                print(f"Found existing CT scan at row {idx}:")
                # print(f"  local_path: {local_path_str}")
                # print(f"  nifti_path: {nifti_path}")
                break
    
    if not nifti_path or not nifti_path.exists():
        print("ERROR: No existing CT scan found in CSV")
        return
    
    # 6. Load NIfTI file
    # print(f"\nLoading NIfTI file: {nifti_path}")
    img_obj = nib.load(str(nifti_path))
    img_data = img_obj.get_fdata()
    
    print(f"Image shape: {img_data.shape}")
    print(f"Image dtype: {img_data.dtype}")
    
    # 7. Create PDF with all slices
    output_dir = Path('/home/dcunhrya/vista_eval/figures/ct_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / 'ct_slices.pdf'
    
    print(f"\nCreating PDF: {pdf_path}")
    
    # Determine number of slices (assuming z-axis is the last dimension)
    num_slices = img_data.shape[2]
    print(f"Number of slices: {num_slices}")
    
    # Create PDF
    with PdfPages(str(pdf_path)) as pdf:
        # Add title page with metadata
        fig_title = plt.figure(figsize=(11, 8.5))
        fig_title.text(0.5, 0.9, 'CT Scan Slices', ha='center', va='top', fontsize=24, fontweight='bold')
        fig_title.text(0.5, 0.8, f'Task: {first_task_name}', ha='center', va='top', fontsize=18)
        
        # NIfTI Path - wrap if too long
        nifti_path_str = str(nifti_path)
        max_chars_per_line = 80
        if len(nifti_path_str) > max_chars_per_line:
            # Split path into multiple lines
            path_lines = []
            for i in range(0, len(nifti_path_str), max_chars_per_line):
                path_lines.append(nifti_path_str[i:i+max_chars_per_line])
        else:
            path_lines = [nifti_path_str]
        
        fig_title.text(0.5, 0.7, 'NIfTI Path:', ha='center', va='top', fontsize=14, fontweight='bold')
        y_pos = 0.65
        for line in path_lines:
            fig_title.text(0.5, y_pos, line, ha='center', va='top', fontsize=11, family='monospace')
            y_pos -= 0.04
        
        # Calculate middle indices for all 3 dimensions
        middle_x = img_data.shape[0] // 2
        middle_y = img_data.shape[1] // 2
        middle_z = img_data.shape[2] // 2
        
        fig_title.text(0.5, y_pos - 0.05, f'Image Shape: {img_data.shape}', ha='center', va='top', fontsize=14)
        fig_title.text(0.5, y_pos - 0.1, f'Number of Slices: {num_slices}', ha='center', va='top', fontsize=14)
        fig_title.text(0.5, y_pos - 0.15, f'Image Dtype: {img_data.dtype}', ha='center', va='top', fontsize=12)
        fig_title.text(0.5, y_pos - 0.22, f'Middle Indices:', ha='center', va='top', fontsize=14, fontweight='bold')
        fig_title.text(0.5, y_pos - 0.27, f'X (Sagittal): {middle_x}/{img_data.shape[0]-1}', ha='center', va='top', fontsize=12)
        fig_title.text(0.5, y_pos - 0.32, f'Y (Coronal): {middle_y}/{img_data.shape[1]-1}', ha='center', va='top', fontsize=12)
        fig_title.text(0.5, y_pos - 0.37, f'Z (Axial): {middle_z}/{img_data.shape[2]-1}', ha='center', va='top', fontsize=12)
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        print(f"Middle indices - X: {middle_x}, Y: {middle_y}, Z: {middle_z}")

        # --- 1. Axial Slice (Z-axis) on its own page ---
        fig_axial = plt.figure(figsize=(10, 10))  # Adjusted size for single page
        axial_slice = img_data[:, :, middle_z]
        print(f"Axial slice shape: {axial_slice.shape}")
        axial_normalized = normalize_slice(axial_slice)
        pil_axial = Image.fromarray(axial_normalized, mode='L')
        pil_axial_resized = pad_to_512(pil_axial)
        axial_resized = np.array(pil_axial_resized)

        plt.imshow(axial_resized, cmap='gray', aspect='auto')
        plt.title(f'Axial (Z-axis)\nIndex: {middle_z}/{img_data.shape[2]-1}', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        pdf.savefig(fig_axial, bbox_inches='tight')
        plt.close(fig_axial)

        # --- 2. Coronal Slice (Y-axis) on its own page ---
        fig_coronal = plt.figure(figsize=(10, 10))
        coronal_slice = img_data[:, middle_y, :]
        print(f"Coronal slice shape: {coronal_slice.shape}")
        coronal_normalized = normalize_slice(coronal_slice)
        pil_coronal = Image.fromarray(coronal_normalized, mode='L')
        pil_coronal_resized = pad_to_512(pil_coronal)
        coronal_resized = np.array(pil_coronal_resized)

        plt.imshow(coronal_resized, cmap='gray', aspect='auto')
        plt.title(f'Coronal (Y-axis)\nIndex: {middle_y}/{img_data.shape[1]-1}', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        pdf.savefig(fig_coronal, bbox_inches='tight')
        plt.close(fig_coronal)

        # --- 3. Sagittal Slice (X-axis) on its own page ---
        fig_sagittal = plt.figure(figsize=(10, 10))
        sagittal_slice = img_data[middle_x, :, :]
        print(f"Sagittal slice shape: {sagittal_slice.shape}")
        sagittal_normalized = normalize_slice(sagittal_slice)
        pil_sagittal = Image.fromarray(sagittal_normalized, mode='L')
        pil_sagittal_resized = pad_to_512(pil_sagittal)
        sagittal_resized = np.array(pil_sagittal_resized)

        plt.imshow(sagittal_resized, cmap='gray', aspect='auto')
        plt.title(f'Sagittal (X-axis)\nIndex: {middle_x}/{img_data.shape[0]-1}', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        pdf.savefig(fig_sagittal, bbox_inches='tight')
        plt.close(fig_sagittal)
        
        # Process each slice
        for slice_idx in range(1):
            # Extract slice
            slice_data = img_data[:, :, slice_idx]
            
            # Normalize slice
            slice_normalized = normalize_slice(slice_data)
            
            # Convert to PIL Image, resize to 512x512, then back to numpy
            pil_img = Image.fromarray(slice_normalized, mode='L')
            pil_img_resized = pad_to_512(pil_img)
            slice_resized = np.array(pil_img_resized)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(slice_resized, cmap='gray', aspect='auto')
            ax.set_title(f'Slice {slice_idx + 1}/{num_slices}', fontsize=12)
            ax.axis('off')
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            if (slice_idx + 1) % 50 == 0:
                print(f"  Processed {slice_idx + 1}/{num_slices} slices...")
    
    print(f"\nPDF created successfully: {pdf_path}")
    print(f"Total slices: {num_slices}")

if __name__ == "__main__":
    create_ct_pdf()
