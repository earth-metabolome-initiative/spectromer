"""Submodule providing spectral datasets."""

from spectromer.datasets.dataset import Dataset
from spectromer.datasets.gnps import GNPS

AVAILABLE_DATASETS = {
    "GNPS": GNPS,
}

__all__ = ["Dataset", "GNPS", "AVAILABLE_DATASETS"]
