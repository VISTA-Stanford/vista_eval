import numpy as np
from google.cloud import storage


def nifti_path_to_blob_path(path_str, bucket_name, prefix):
    """
    Convert a nifti_path value to the GCS blob path (same logic as download_subsampled_ct.py).
    path_str may be local path, bucket-relative path, or filename.
    Returns the blob path string (relative to bucket).
    """
    if not path_str or (isinstance(path_str, float) and np.isnan(path_str)):
        return None
    path_str = str(path_str).strip()
    if not path_str:
        return None
    # Remove /mnt/ prefix if present
    if path_str.startswith("/mnt/"):
        path_str = path_str[5:]
    # Remove bucket name prefix if present
    if path_str.startswith(f"{bucket_name}/"):
        path_str = path_str[len(bucket_name) + 1:]
    # Already a full bucket-relative path
    if path_str.startswith(prefix):
        return path_str
    # Build from path parts
    parts = path_str.split("/")
    filename = parts[-1]
    if not filename.endswith(".nii.gz"):
        if len(parts) >= 2:
            filename_no_ext = parts[-1].replace(".zip", "")
            bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
        else:
            bucket_filename = filename if filename.endswith(".nii.gz") else f"{filename}.nii.gz"
    else:
        bucket_filename = filename
    return f"{prefix}/{bucket_filename}"


def check_nifti_exists_in_bucket(path_str, bucket_name, prefix, bucket):
    """Return True if the nifti_path exists in the GCS bucket (same check as download_subsampled_ct.py)."""
    blob_path = nifti_path_to_blob_path(path_str, bucket_name, prefix)
    if not blob_path:
        return False
    try:
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception:
        return False


def filter_person_ids_by_bucket_existence(
    person_ids,
    path_pairs,
    bucket_name,
    prefix,
):
    """
    Filter person_ids to only those who have at least one nifti_path that exists in the GCS bucket.
    path_pairs: iterable of (person_id, nifti_path) with non-null, non-empty nifti_path.
    Returns set of person_ids (int and str) for which at least one path exists in bucket.
    """
    if not person_ids or not path_pairs:
        return set()
    pairs = []
    seen = set()
    for pid, path in path_pairs:
        if path is None or (isinstance(path, float) and np.isnan(path)):
            continue
        path = str(path).strip()
        if not path:
            continue
        key = (str(pid), path)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((pid, path))
    if not pairs:
        return set()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"  Warning: Could not create GCS client for bucket existence check: {e}")
        return set()
    path_to_exists = {}
    for _pid, path in pairs:
        if path not in path_to_exists:
            path_to_exists[path] = check_nifti_exists_in_bucket(
                path, bucket_name, prefix, bucket
            )
    result = set()
    for pid, path in pairs:
        if path_to_exists.get(path, False):
            result.add(pid)
            try:
                result.add(int(pid))
                result.add(str(pid))
            except (ValueError, TypeError):
                result.add(pid)
                result.add(str(pid))
    return result