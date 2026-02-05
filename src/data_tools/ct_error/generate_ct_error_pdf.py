#!/usr/bin/env python3
"""
Generate PDFs from CT scans.
Takes a list of CT scan filenames (nii.gz), loads them from GCP bucket or local storage,
and creates a PDF with the filename and middle slices from all 3 dimensions.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from google.cloud import storage

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from vista_run.utils.utils_inference import normalize_slice


def construct_ct_path(nii_filename, bucket_name='su-vista-uscentral1', 
                      prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                      download_base_dir='/home/dcunhrya/downloaded_ct_scans'):
    """
    Construct the full path to a CT scan file.
    
    Args:
        nii_filename: Just the nii.gz filename (e.g., 'folder__filename.nii.gz')
        bucket_name: GCP bucket name
        prefix: GCP bucket prefix
        download_base_dir: Local download directory base
    
    Returns:
        tuple: (local_path, gcp_blob_path, bucket_name)
    """
    # Local path: download_base_dir/prefix/nii_filename
    local_path = Path(download_base_dir) / prefix / nii_filename
    
    # GCP blob path: prefix/nii_filename
    gcp_blob_path = f"{prefix}/{nii_filename}"
    
    return local_path, gcp_blob_path, bucket_name


def load_ct_scan(nii_filename, bucket_name='su-vista-uscentral1',
                 prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                 download_base_dir='/home/dcunhrya/downloaded_ct_scans'):
    """
    Load a CT scan from local storage or download from GCP if needed.
    
    Args:
        nii_filename: Just the nii.gz filename
        bucket_name: GCP bucket name
        prefix: GCP bucket prefix
        download_base_dir: Local download directory base
    
    Returns:
        tuple: (img_data, img_obj) or (None, None) if file not found
    """
    local_path, gcp_blob_path, bucket_name = construct_ct_path(
        nii_filename, bucket_name, prefix, download_base_dir
    )
    
    # Check if file exists locally
    if local_path.exists():
        print(f"  Loading from local: {local_path}")
        img_obj = nib.load(str(local_path))
        img_data = img_obj.get_fdata()
        return img_data, img_obj
    
    # Try to download from GCP
    print(f"  File not found locally, attempting to download from GCP: {gcp_blob_path}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcp_blob_path)
        
        if blob.exists():
            # Create local directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            print(f"  Downloading from GCP...")
            blob.download_to_filename(str(local_path))
            
            # Load the downloaded file
            img_obj = nib.load(str(local_path))
            img_data = img_obj.get_fdata()
            return img_data, img_obj
        else:
            print(f"  ERROR: File not found in GCP bucket: {gcp_blob_path}")
            return None, None
    except Exception as e:
        print(f"  ERROR: Failed to download from GCP: {e}")
        return None, None


def create_ct_pdf(ct_filenames, output_pdf_path, bucket_name='su-vista-uscentral1',
                  prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                  download_base_dir='/home/dcunhrya/downloaded_ct_scans'):
    """
    Create a PDF with CT scan visualizations.
    
    Args:
        ct_filenames: List of nii.gz filenames (e.g., ['folder__file1.nii.gz', 'folder__file2.nii.gz'])
        output_pdf_path: Path to output PDF file
        bucket_name: GCP bucket name
        prefix: GCP bucket prefix
        download_base_dir: Local download directory base
    """
    if not ct_filenames:
        print("ERROR: No CT filenames provided")
        return
    
    # Ensure output directory exists
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating PDF with {len(ct_filenames)} CT scans...")
    print(f"Output PDF: {output_pdf_path}")
    
    with PdfPages(str(output_pdf_path)) as pdf:
        for idx, nii_filename in enumerate(ct_filenames, 1):
            print(f"\n[{idx}/{len(ct_filenames)}] Processing: {nii_filename}")
            
            # Load CT scan
            img_data, img_obj = load_ct_scan(
                nii_filename, bucket_name, prefix, download_base_dir
            )
            
            if img_data is None:
                print(f"  SKIPPING: Could not load {nii_filename}")
                continue
            
            print(f"  Image shape: {img_data.shape}")
            
            # Calculate middle indices for all 3 dimensions
            middle_x = img_data.shape[0] // 2
            middle_y = img_data.shape[1] // 2
            middle_z = img_data.shape[2] // 2
            
            # Extract middle slices
            axial_slice = img_data[:, :, middle_z]      # Z-axis (axial)
            coronal_slice = img_data[:, middle_y, :]    # Y-axis (coronal)
            sagittal_slice = img_data[middle_x, :, :]   # X-axis (sagittal)
            
            # Normalize slices
            axial_normalized = normalize_slice(axial_slice)
            coronal_normalized = normalize_slice(coronal_slice)
            sagittal_normalized = normalize_slice(sagittal_slice)
            
            # Create figure with 3 subplots arranged horizontally or in a grid
            # Use a single page with 3 small images
            fig = plt.figure(figsize=(11, 4))  # Wide figure to fit 3 images
            
            # Add title with filename and image shape
            title_text = f"{nii_filename}\nShape: {img_data.shape}"
            fig.suptitle(title_text, fontsize=10, fontweight='bold', y=0.95)
            
            # Axial (Z-axis) - left
            ax1 = plt.subplot(1, 3, 1)
            ax1.imshow(axial_normalized, cmap='gray', aspect='auto')
            ax1.set_title(f'Axial (Z={middle_z})', fontsize=9)
            ax1.axis('off')
            
            # Coronal (Y-axis) - middle
            ax2 = plt.subplot(1, 3, 2)
            ax2.imshow(coronal_normalized, cmap='gray', aspect='auto')
            ax2.set_title(f'Coronal (Y={middle_y})', fontsize=9)
            ax2.axis('off')
            
            # Sagittal (X-axis) - right
            ax3 = plt.subplot(1, 3, 3)
            ax3.imshow(sagittal_normalized, cmap='gray', aspect='auto')
            ax3.set_title(f'Sagittal (X={middle_x})', fontsize=9)
            ax3.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for title
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            print(f"  Added to PDF: {nii_filename}")
    
    print(f"\nPDF created successfully: {output_pdf_path}")
    print(f"Total CT scans processed: {len(ct_filenames)}")


def read_filenames_from_file(file_path):
    """
    Read CT scan filenames from a text file (one per line).
    
    Args:
        file_path: Path to text file with one filename per line
    
    Returns:
        list: List of filenames (stripped of whitespace, empty lines skipped)
    """
    filenames = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                filenames.append(line)
    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate PDF from CT scans with middle slices from all 3 dimensions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From command line arguments:
  python generate_ct_pdf.py file1.nii.gz file2.nii.gz -o output.pdf
  
  # From a text file (one filename per line):
  python generate_ct_pdf.py -f filenames.txt -o output.pdf
  
  # With custom GCP settings:
  python generate_ct_pdf.py -f filenames.txt -o output.pdf --bucket my-bucket --prefix my/prefix
        """
    )
    
    # Input options
    parser.add_argument('-f', '--file', type=str,
                       default='/home/dcunhrya/vista_eval/src/scratch/ct_info/ct_filenames.txt',
                       help='Text file with CT scan filenames (one per line) (default: %(default)s)')
    parser.add_argument('filenames', nargs='*',
                       help='CT scan filenames (nii.gz) as command-line arguments (overrides -f/--file if provided)')
    
    # Output option
    parser.add_argument('-o', '--output', type=str,
                       default='/home/dcunhrya/vista_eval/figures/ct_info/ct_error_scans.pdf',
                       help='Output PDF path (default: %(default)s)')
    
    # GCP options
    parser.add_argument('--bucket', type=str, default='su-vista-uscentral1',
                       help='GCP bucket name (default: %(default)s)')
    parser.add_argument('--prefix', type=str,
                       default='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                       help='GCP bucket prefix (default: %(default)s)')
    parser.add_argument('--download-dir', type=str,
                       default='/home/dcunhrya/downloaded_ct_scans',
                       help='Local download directory (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Get filenames - prioritize command-line arguments over file
    if args.filenames:
        ct_filenames = args.filenames
        print(f"Using {len(ct_filenames)} filenames from command line")
    else:
        if not Path(args.file).exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        ct_filenames = read_filenames_from_file(args.file)
        print(f"Read {len(ct_filenames)} filenames from: {args.file}")
    
    if not ct_filenames:
        print("ERROR: No CT filenames provided")
        sys.exit(1)
    
    # Create PDF
    create_ct_pdf(
        ct_filenames,
        args.output,
        bucket_name=args.bucket,
        prefix=args.prefix,
        download_base_dir=args.download_dir
    )
