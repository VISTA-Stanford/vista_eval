"""
CSV I/O utilities for VISTA data tools.
"""
from pathlib import Path

import pandas as pd


def append_to_csv(file_path: Path, data_list: list, header: bool = None) -> None:
    """
    Append data to CSV without rewriting the whole file.

    Args:
        file_path: Path to the CSV file.
        data_list: List of dicts to append (will be converted to DataFrame).
        header: If True, write header. If False, omit. If None, write header only when file doesn't exist.
    """
    new_df = pd.DataFrame(data_list)
    if header is None:
        header = not file_path.exists()
    new_df.to_csv(file_path, mode='a', index=False, header=header)
