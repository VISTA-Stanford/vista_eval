#!/usr/bin/env python3
"""
Extract CT scan filenames from log files that contain NIfTI loading errors.
Finds 'Error loading NIfTI from nifti_path' messages and extracts the filename.
"""

import argparse
import re
from pathlib import Path


def extract_ct_filenames_from_log(log_path, output_path):
    """
    Extract CT filenames from log file that contain NIfTI loading errors.
    
    Args:
        log_path: Path to the log file
        output_path: Path to output text file (will be overwritten)
    """
    log_file = Path(log_path)
    output_file = Path(output_path)
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    # Pattern to match: "Error loading NIfTI from nifti_path <path>: <error message>"
    # We want to extract the path part
    pattern = r'Error loading NIfTI from nifti_path\s+(.+?):'
    
    ct_filenames = set()  # Use set to avoid duplicates
    
    print(f"Reading log file: {log_path}")
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                # Extract the full path
                full_path = match.group(1).strip()
                
                # Get the last part of the path (filename)
                filename = full_path.split('/')[-1]
                
                # Only add if it ends with .nii.gz
                if filename.endswith('.nii.gz'):
                    ct_filenames.add(filename)
                    # print(f"  Found error: {filename}")
    
    # Sort for consistent output
    ct_filenames = sorted(ct_filenames)
    
    # Write to output file (overwrite mode)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for filename in ct_filenames:
            f.write(f"{filename}\n")
    
    print(f"\nExtracted {len(ct_filenames)} unique CT filenames with errors")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract CT scan filenames from log files that contain NIfTI loading errors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default log file and output:
  python collect_ct_error.py
  
  # Specify custom log file:
  python collect_ct_error.py -l /path/to/logfile.log
  
  # Specify both log file and output:
  python collect_ct_error.py -l /path/to/logfile.log -o /path/to/output.txt
        """
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default='/home/dcunhrya/vista_eval/logs/20260125_054108/model_1_OctoMed_OctoMed-7B.log',
        help='Path to log file (default: %(default)s)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/dcunhrya/vista_eval/src/scratch/ct_error/ct_filenames.txt',
        help='Path to output text file (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    extract_ct_filenames_from_log(args.log, args.output)
