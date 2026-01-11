import numpy as np


def safe_scalar(x):
    try:
        arr = np.array(x)
        val = np.nanmean(arr)
        return float(val) if np.isfinite(val) else 0.0
    except Exception:
        return 0.0
