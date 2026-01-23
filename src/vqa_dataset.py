import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
from vista_run.utils.utils_inference import pad_to_512, normalize_slice

class PromptDataset(Dataset):
    def __init__(self, df, prompt_col='dynamic_prompt', add_options=False, experiment='axial_1_image'):
        """
        Args:
            df: Dataframe containing the data.
            prompt_col: The column name to use for the text prompt.
            add_options: Whether to append options to the prompt.
            experiment: Experiment type - 'no_image', 'axial_1_image', 'all_image', or 'axial_all_image'
        """
        self.df = df.reset_index(drop=True)
        self.prompt_col = prompt_col
        self.add_options = add_options
        self.experiment = experiment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Handle Prompt
        question = str(row[self.prompt_col])
        
        if self.add_options and 'options' in row and pd.notna(row['options']):
            question = f"{question} Options: {row['options']}"

        # 2. Handle Image based on experiment type
        img = None
        image_path = row.get('image_path', None)
        
        # Skip image loading for 'no_image' experiment
        if self.experiment == 'no_image':
            img = None
        else:
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
                            
                            # Handle different experiment types
                            if self.experiment == 'axial_1_image':
                                # Extract axial middle slice (3rd dimension)
                                axial_middle_index = img_data.shape[2] // 2
                                axial_slice = img_data[:, :, axial_middle_index]
                                axial_slice = normalize_slice(axial_slice)
                                img = Image.fromarray(axial_slice, mode='L')
                                img = pad_to_512(img)
                                print(f"[IMAGE] Using NIfTI axial_1_image from local_path: {local_path} (index: {row.get('index', idx)})")
                            
                            elif self.experiment == 'all_image':
                                # Extract middle index of each of the 3 dimensions
                                img_list = []
                                # Dimension 0 (sagittal)
                                if len(img_data.shape) > 0:
                                    sagittal_middle = img_data.shape[0] // 2
                                    sagittal_slice = img_data[sagittal_middle, :, :]
                                    sagittal_slice = normalize_slice(sagittal_slice)
                                    img_list.append(pad_to_512(Image.fromarray(sagittal_slice, mode='L')))
                                # Dimension 1 (coronal)
                                if len(img_data.shape) > 1:
                                    coronal_middle = img_data.shape[1] // 2
                                    coronal_slice = img_data[:, coronal_middle, :]
                                    coronal_slice = normalize_slice(coronal_slice)
                                    img_list.append(pad_to_512(Image.fromarray(coronal_slice, mode='L')))
                                # Dimension 2 (axial)
                                if len(img_data.shape) > 2:
                                    axial_middle = img_data.shape[2] // 2
                                    axial_slice = img_data[:, :, axial_middle]
                                    axial_slice = normalize_slice(axial_slice)
                                    img_list.append(pad_to_512(Image.fromarray(axial_slice, mode='L')))
                                img = img_list if img_list else None
                                print(f"[IMAGE] Using NIfTI all_image from local_path: {local_path} (index: {row.get('index', idx)})")
                            
                            elif self.experiment == 'axial_all_image':
                                # Extract 10 images at 0, 0.1, 0.2, ... 1.0 * len(image.shape[2])
                                # Using 0.0, 0.1, ..., 0.9 to get 10 evenly spaced slices
                                if len(img_data.shape) > 2:
                                    depth = img_data.shape[2]
                                    img_list = []
                                    for i in range(10):  # 10 total images
                                        # Calculate position: 0.0, 0.1, 0.2, ..., 0.9
                                        position = i * 0.1
                                        index = int(position * (depth - 1))
                                        if index >= depth:
                                            index = depth - 1
                                        axial_slice = img_data[:, :, index]
                                        axial_slice = normalize_slice(axial_slice)
                                        img_list.append(pad_to_512(Image.fromarray(axial_slice, mode='L')))
                                    img = img_list
                                    print(f"[IMAGE] Using NIfTI axial_all_image from local_path: {local_path} (index: {row.get('index', idx)})")
                                else:
                                    img = None
                        # else:
                        #     print(f"[IMAGE] local_path present but file not found: {local_path} -> {nifti_path} (index: {row.get('index', idx)}) - using text-only")
                    except Exception as e:
                        print(f"Error loading NIfTI from local_path {local_path}: {e}")
                        # Continue with text-only if image loading fails

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