import os
import pandas as pd
import csv
import sys
from pathlib import Path

# Increase the limit to handle very large clinical text fields
# We set it to the maximum allowed by the system
csv.field_size_limit(sys.maxsize)

def subsample_csvs(base_path, target_n=10):
    base_dir = Path(base_path)
    
    for csv_path in base_dir.rglob('*.csv'):
        if csv_path.stem.endswith('_subsampled'):
            continue
            
        print(f"Processing: {csv_path.name}")
        
        try:
            # Using sep=None with engine='python' to auto-detect Tab vs Comma
            # We also pass low_memory=False to handle mixed types in large files
            df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='warn')
            
            # Find the label column (case-insensitive)
            label_col = next((c for c in df.columns if c.lower() == 'label'), None)
            
            if label_col is None:
                print(f"  Skipping: No 'label' column found in {csv_path.name}")
                print(f"  Available columns: {list(df.columns)}")
                continue
                
            # Perform stratified sampling
            subsampled_df = df.groupby(label_col, group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), target_n), random_state=42)
            )
            
            new_path = csv_path.with_name(f"{csv_path.stem}_subsampled.csv")
            
            # Save the result
            subsampled_df.to_csv(new_path, index=False)
            print(f"  Saved: {new_path.name} ({len(subsampled_df)} rows)")
            
        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")

if __name__ == "__main__":
    PATH = '/home/dcunhrya/vista_bench'
    subsample_csvs(PATH)