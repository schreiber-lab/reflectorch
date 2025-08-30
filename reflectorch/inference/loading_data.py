from pathlib import Path
import numpy as np

def load_mft_data(filepath):
    """
    Load q, reflectivity, reflectivity error, and q-resolution from an .mft file.
    
    Parameters:
        filepath (str or Path): Path to the .mft file

    Returns:
        q, refl, refl_err, q_res : np.ndarray
    """
    filepath = Path(filepath)
    
    with filepath.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    start_idx = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith('q') and 'q_res' in line
    ) + 1

    data = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                data.append([float(p.replace('E', 'e')) for p in parts])
            except ValueError:
                continue

    if not data:
        raise ValueError(f"No valid data found in {filepath}")

    data_array = np.array(data)
    return data_array.T