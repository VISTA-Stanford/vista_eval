import os
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset

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
        
        if pd.notna(image_path) and os.path.exists(str(image_path)):
            try:
                with Image.open(image_path) as PIL_img:
                    PIL_img.load()
                    img = pad_to_512(PIL_img.copy())
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")

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

def pad_to_512(img):
    """Pads image to 512x512 with right/bottom padding."""
    w, h = img.size
    size = 512
    if w >= size and h >= size:
        return img
    pad_w = max(0, size - w)
    pad_h = max(0, size - h)
    return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)

def prompt_collate(batch):
    """Returns the batch as a list of dictionaries."""
    return batch