"""Submodule retriening the GNPS dataset."""

import os
from typing import Iterable, List
from downloaders import BaseDownloader
from matchms import Spectrum
from matchms.importing import load_from_mgf
from skfp.bases import BaseFingerprintTransformer
from spectromer.datasets.dataset import Dataset


class GNPS(Dataset):
    """GNPS dataset."""

    def __init__(
        self,
        fingerprints: List[BaseFingerprintTransformer],
        batch_size: int = 1024,
        number_of_peaks: int = 512,
        directory: str = "datasets",
        verbose: bool = True,
    ):
        """Initialize the dataset."""
        super().__init__(
            fingerprints=fingerprints,
            directory=directory,
            batch_size=batch_size,
            number_of_peaks=number_of_peaks,
            verbose=verbose,
        )

    @property
    def name(self) -> str:
        return "GNPS"

    def load(self) -> Iterable[Spectrum]:
        """Load the dataset."""
        downloader = BaseDownloader(process_number=1)
        downloader.download(
            "https://external.gnps2.org/processed_gnps_data/matchms.mgf",
            os.path.join(self.directory, "matchms.mgf"),
        )
        return load_from_mgf(os.path.join(self.directory, "matchms.mgf"))
