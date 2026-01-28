import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
import io
import tempfile
from vista_run.utils.utils_inference import pad_to_512, pad_to_size, normalize_slice

def norm(ct_vol: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Window and normalize CT imaging Houndsfield values to values 0 - 255."""
    ct_vol = np.clip(ct_vol, min_val, max_val)  # Clip the imaging value range
    ct_vol = ct_vol.astype(np.float32)
    ct_vol -= min_val
    ct_vol /= (max_val - min_val)  # Norm to values between 0 - 1.0
    ct_vol *= 255.0  # Norm to values been 0 - 255.0
    return ct_vol

def window(ct_vol: np.ndarray) -> np.ndarray:
    """Window CT slice imaging with three windows (wide, mediastinum(chest), brain).
    Imaging will appear color when visualized, RGB channels contain different
    representations of the data.
    """
    window_clips = [(-1024, 1024), (-135, 215), (0, 80)]
    return np.stack([norm(ct_vol, clip[0], clip[1]) for clip in window_clips], axis=-1)

class PromptDataset(Dataset):
    def __init__(self, df, prompt_col='dynamic_prompt', add_options=False, experiment='axial_1_image', storage_client=None, model_type=None):
        """
        Args:
            df: Dataframe containing the data.
            prompt_col: The column name to use for the text prompt.
            add_options: Whether to append options to the prompt.
            experiment: Experiment type - 'no_image', 'axial_1_image', 'all_image', 'axial_all_image', 'sagittal_all_image', or 'no_timeline'
            storage_client: GCP Storage client for loading NIfTI files from bucket
            model_type: Model type string (e.g., 'gemma3') to determine preprocessing
        """
        self.df = df.reset_index(drop=True)
        self.prompt_col = prompt_col
        self.add_options = add_options
        self.experiment = experiment
        self.storage_client = storage_client
        self.bucket_name = 'su-vista-uscentral1'
        self.model_type = model_type
        self.is_gemma = model_type is not None and 'gemma' in model_type.lower()
        
        # Determine image size based on model type
        self.target_size = 448 if self.is_gemma else 512
    
    def _process_ct_slice(self, ct_slice):
        """
        Process a CT slice: apply windowing for gemma models, normalize, convert to PIL Image, and resize.
        
        Args:
            ct_slice: numpy array of CT slice data
            
        Returns:
            PIL Image object
        """
        if self.is_gemma:
            # Apply windowing function for gemma models
            windowed_slice = window(ct_slice)
            # Round slice voxels to nearest integer number
            windowed_slice = np.round(windowed_slice, 0).astype(np.uint8)
            # Convert to PIL Image (RGB mode since windowing creates 3 channels)
            pil_img = Image.fromarray(windowed_slice, mode='RGB')
        else:
            # Standard normalization for non-gemma models
            normalized_slice = normalize_slice(ct_slice)
            pil_img = Image.fromarray(normalized_slice, mode='L')
        
        # Resize to target size (448 for gemma, 512 for others)
        return pad_to_size(pil_img, self.target_size)

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
                        img = pad_to_size(PIL_img.copy(), self.target_size)
                    print(f"[IMAGE] Using image from image_path: {image_path} (index: {row.get('index', idx)})")
                except Exception as e:
                    print(f"Error loading image at {image_path}: {e}")
            
            # If no image from image_path, check for nifti_path (NIfTI files in GCP bucket)
            if img is None:
                nifti_path = row.get('nifti_path', None)
                if pd.notna(nifti_path) and isinstance(nifti_path, str) and self.storage_client is not None:
                    try:
                        # Remove '/mnt/' prefix if present
                        if nifti_path.startswith('/mnt/'):
                            nifti_path = nifti_path[5:]  # Remove '/mnt/'
                        
                        # Remove bucket name prefix if present (path should be relative to bucket)
                        if nifti_path.startswith(f'{self.bucket_name}/'):
                            nifti_path = nifti_path[len(self.bucket_name) + 1:]  # Remove 'bucket_name/'
                        
                        # Load NIfTI file from GCP bucket
                        import nibabel as nib
                        bucket = self.storage_client.bucket(self.bucket_name)
                        blob = bucket.blob(nifti_path)
                        
                        # Download to memory
                        nifti_bytes = blob.download_as_bytes()
                        
                        # Load from bytes using nibabel
                        # Use a temporary file since nibabel.load() may not accept BytesIO directly
                        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                            tmp_file.write(nifti_bytes)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            img_obj = nib.load(tmp_file_path)
                            img_data = img_obj.get_fdata()
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                        
                        # Handle different experiment types
                        if self.experiment == 'axial_1_image':
                            # Extract axial middle slice (3rd dimension)
                            axial_middle_index = img_data.shape[2] // 2
                            axial_slice = img_data[:, :, axial_middle_index]
                            img = self._process_ct_slice(axial_slice)
                            # print(f"[IMAGE] Using NIfTI axial_1_image from nifti_path: {nifti_path} (index: {row.get('index', idx)})")
                            
                        elif self.experiment == 'all_image':
                            # Extract middle index of each of the 3 dimensions
                            img_list = []
                            # Dimension 0 (sagittal)
                            if len(img_data.shape) > 0:
                                sagittal_middle = img_data.shape[0] // 2
                                sagittal_slice = img_data[sagittal_middle, :, :]
                                img_list.append(self._process_ct_slice(sagittal_slice))
                            # Dimension 1 (coronal)
                            if len(img_data.shape) > 1:
                                coronal_middle = img_data.shape[1] // 2
                                coronal_slice = img_data[:, coronal_middle, :]
                                img_list.append(self._process_ct_slice(coronal_slice))
                            # Dimension 2 (axial)
                            if len(img_data.shape) > 2:
                                axial_middle = img_data.shape[2] // 2
                                axial_slice = img_data[:, :, axial_middle]
                                img_list.append(self._process_ct_slice(axial_slice))
                            img = img_list if img_list else None
                            # print(f"[IMAGE] Using NIfTI all_image from nifti_path: {nifti_path} (index: {row.get('index', idx)})")
                            
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
                                    img_list.append(self._process_ct_slice(axial_slice))
                                img = img_list
                                # print(f"[IMAGE] Using NIfTI axial_all_image from nifti_path: {nifti_path} (index: {row.get('index', idx)})")
                            else:
                                img = None
                                
                        elif self.experiment == 'sagittal_all_image':
                            # Extract 10 images at 0, 0.1, 0.2, ... 1.0 * len(image.shape[0])
                            # Using 0.0, 0.1, ..., 0.9 to get 10 evenly spaced slices
                            if len(img_data.shape) > 0:
                                width = img_data.shape[0]
                                img_list = []
                                for i in range(10):  # 10 total images
                                    # Calculate position: 0.0, 0.1, 0.2, ..., 0.9
                                    position = i * 0.1
                                    index = int(position * (width - 1))
                                    if index >= width:
                                        index = width - 1
                                    sagittal_slice = img_data[index, :, :]
                                    img_list.append(self._process_ct_slice(sagittal_slice))
                                img = img_list
                                # print(f"[IMAGE] Using NIfTI sagittal_all_image from nifti_path: {nifti_path} (index: {row.get('index', idx)})")
                            else:
                                img = None
                                
                        elif self.experiment == 'no_timeline':
                            # Extract 50 evenly spaced axial slices (3rd dimension)
                            # This experiment is for when no patient timeline is provided
                            if len(img_data.shape) > 2:
                                depth = img_data.shape[2]
                                img_list = []
                                num_slices = 100
                                for i in range(num_slices):
                                    # Calculate position: 0.0, 1/(num_slices-1), 2/(num_slices-1), ..., 1.0
                                    if num_slices > 1:
                                        position = i / (num_slices - 1)
                                    else:
                                        position = 0.0
                                    index = int(position * (depth - 1))
                                    if index >= depth:
                                        index = depth - 1
                                    axial_slice = img_data[:, :, index]
                                    img_list.append(self._process_ct_slice(axial_slice))
                                img = img_list
                                # print(f"[IMAGE] Using NIfTI no_timeline (50 slices) from nifti_path: {nifti_path} (index: {row.get('index', idx)})")
                            else:
                                img = None
                    except Exception as e:
                        print(f"Error loading NIfTI from nifti_path {nifti_path}: {e}")
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