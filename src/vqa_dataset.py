import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
from vista_run.utils.utils_inference import pad_to_512, normalize_slice

class PromptDataset(Dataset):
    def __init__(self, df, prompt_col='dynamic_prompt', add_options=False):
        """
        Args:
            df: Dataframe containing the data.
            prompt_col: The column name to use for the text prompt.
            add_options: Whether to append options to the prompt.
        """
        self.df = df.reset_index(drop=True)
        self.prompt_col = prompt_col
        self.add_options = add_options

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Handle Prompt
        question = str(row[self.prompt_col])
        
        if self.add_options and 'options' in row and pd.notna(row['options']):
            question = f"{question} Options: {row['options']}"

        # 2. Handle Image (Robust for text-only or future image use)
        img = None
        image_path = row.get('image_path', None)
        
        # First check for image_path (existing behavior)
        if pd.notna(image_path) and os.path.exists(str(image_path)):
            try:
                with Image.open(image_path) as PIL_img:
                    PIL_img.load()
                    img = pad_to_512(PIL_img.copy())
                print(f"[IMAGE] Using image from image_path: {image_path} (index: {row.get('index', idx)})")
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
        
        # If no image from image_path, check for local_path (NIfTI files)
        if img is None:
            local_path = row.get('local_path', None)
            if pd.notna(local_path) and isinstance(local_path, str):
                try:
                    # Transform local_path to actual file path (following download_subsampled_ct.py logic)
                    parts = local_path.split('/')
                    filename_no_ext = parts[-1].replace('.zip', '')
                    bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
                    
                    # Construct the actual file path
                    download_base = Path('/home/dcunhrya/downloaded_ct_scans')
                    prefix = 'chaudhari_lab/ct_data/ct_scans/vista/nov25'
                    nifti_path = download_base / prefix / bucket_filename
                    
                    # Check if file exists and load with nibabel
                    if nifti_path.exists():
                        import nibabel as nib
                        img_obj = nib.load(str(nifti_path))
                        img_data = img_obj.get_fdata()
                        
                        # Extract axial middle slice
                        axial_middle_index = img_data.shape[2] // 2
                        axial_slice = img_data[:, :, axial_middle_index]
                        axial_slice = normalize_slice(axial_slice)
                        
                        # Convert to PIL Image
                        img = Image.fromarray(axial_slice, mode='L')  # L mode for grayscale
                        img = pad_to_512(img)
                        print(f"[IMAGE] Using NIfTI image from local_path: {local_path} -> {nifti_path} (index: {row.get('index', idx)})")
                    # else:
                    #     print(f"[IMAGE] local_path present but file not found: {local_path} -> {nifti_path} (index: {row.get('index', idx)}) - using text-only")
                except Exception as e:
                    print(f"Error loading NIfTI from local_path {local_path}: {e}")
                    # Continue with text-only if image loading fails

        # Log if no image is present (text-only mode)
        # Only log if we haven't already logged about local_path
        if img is None:
            local_path = row.get('local_path', None)
            image_path = row.get('image_path', None)
            # Only log if neither image_path nor local_path were present/valid
            # if (pd.isna(local_path) or not isinstance(local_path, str)) and (pd.isna(image_path) or not os.path.exists(str(image_path))):
            #     print(f"[IMAGE] No image present - text-only mode (index: {row.get('index', idx)})")

        # 3. Build Item
        item = {
            "index": row.get("index", idx),
            "question": question,
            "image_path": image_path,
            "image": img,
            "options": row.get("options", None),
            # Pass the raw row so the Orchestrator can access all original metadata for saving
            "raw_row": row 
        }

        return item

def prompt_collate(batch):
    """Returns the batch as a list of dictionaries."""
    return batch