"""
Generic helpers.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def index_to_latlon_coords(
    idx: int, latitude: np.ndarray, longitude: np.ndarray
) -> Tuple[float, float]:
    """
    Convert flattened spatial index to (lat, lon) coordinate.
    """
    nlon = longitude.shape[0]
    lat_idx = idx // nlon
    lon_idx = idx % nlon
    return float(latitude[lat_idx]), float(longitude[lon_idx])