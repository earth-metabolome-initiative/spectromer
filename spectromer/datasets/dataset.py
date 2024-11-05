"""Submodule defining the spectral dataset abstract class."""

from abc import abstractmethod
import os
from copy import copy
from typing import List, Dict, Optional, Iterable, Tuple, Union
from tqdm.auto import tqdm
from keras.api.utils import Sequence
import numpy as np
from dict_hash import sha256
from matchms import Spectrum
from matchms.filtering import (
    default_filters,
    add_parent_mass,
    normalize_intensities,
    reduce_to_number_of_peaks,
    select_by_mz,
    select_by_intensity,
    require_minimum_number_of_peaks,
)
from skfp.bases import BaseFingerprintTransformer


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Jaccard similarity between two arrays.

    Parameters
    ----------
    a : np.ndarray
        The first array.
    b : np.ndarray
        The second array.

    Returns
    -------
    float
        The Jaccard similarity between the two arrays.
    """
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b)) + 1e-6
    return intersection / union


def normalize_charge(charge: Optional[Union[int, float, str]]) -> float:
    """Normalize the charge of a precursor ion.

    Parameters
    ----------
    charge : str
        The charge of the precursor ion.

    Returns
    -------
    float
        The normalized charge of the precursor ion.
    """
    if charge is None:
        return 0.0

    if isinstance(charge, (int, float)):
        return float(charge)

    contains_minus: bool = "-" in charge

    normalized_charge: str = charge.replace("+", "").replace("-", "")

    if contains_minus:
        return -float(normalized_charge)

    return float(normalized_charge)


def extract_mz_features(spectrum: Spectrum) -> np.ndarray:
    """Extract MZ features from a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum to extract MZ features from.

    Returns
    -------
    np.ndarray
        The extracted MZ features.
    """
    return np.array(
        [
            normalize_charge(spectrum.get("charge", 0.0)) / 10.0,
            spectrum.get("precursor_mz", spectrum.get("parent_mass", 0.0)) / 2000.0,
            spectrum.get("parent_mass", spectrum.get("precursor_mz", 0.0)) / 2000.0,
        ]
    )


def spectrum_processing(spectrum: Spectrum, number_of_peaks: int) -> Optional[Spectrum]:
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    spectrum = default_filters(spectrum)
    spectrum = add_parent_mass(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = normalize_intensities(spectrum)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=number_of_peaks)
    spectrum = select_by_mz(spectrum, mz_from=0, mz_to=2000)
    spectrum = require_minimum_number_of_peaks(spectrum)
    return spectrum


class Dataset(Sequence):
    """Abstract class for spectral datasets."""

    def __init__(
        self,
        fingerprints: List[BaseFingerprintTransformer],
        directory: str = "datasets",
        batch_size: int = 1024,
        number_of_peaks: int = 512,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """Initialize the dataset."""
        super().__init__(
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
        )
        self._directory = directory
        self._batch_size = batch_size
        self._verbose = verbose
        self._number_of_peaks = number_of_peaks
        self._random_state = random_state
        self._fingerprints = fingerprints

        # Load the dataset
        if not os.path.exists(self.preprocessed_directory):
            self._preprocess()

        # Load the preprocessed dataset
        (
            self._padded_spectra,
            self._smiles_ids,
            self._spectral_features,
            self._fingerprints_data,
        ) = self._load_preprocessed()

        self._indices = np.arange(len(self._padded_spectra))

    def copy(self) -> "Dataset":
        """Copy the dataset."""
        return copy(self)

    # pylint: disable=protected-access
    def split(self, train_size: float = 0.8) -> Tuple["Dataset", "Dataset"]:
        """Split the dataset into a training and a validation set."""
        train_size = int(train_size * len(self._indices))
        np.random.seed(self._random_state)
        np.random.shuffle(self._indices)

        train_indices = np.array(self._indices[:train_size])
        val_indices = np.array(self._indices[train_size:])

        train_dataset = self.copy()
        train_dataset._indices = train_indices

        val_dataset = self.copy()
        val_dataset._indices = val_indices

        return train_dataset, val_dataset

    @property
    def directory(self) -> str:
        """Directory where the dataset is stored."""
        return os.path.join(self._directory, self.name)

    @property
    def preprocessed_directory(self) -> str:
        """Directory where the preprocessed dataset is stored."""
        return os.path.join(self.directory, self.parameters_hash)

    @property
    def number_of_peaks(self) -> int:
        """Number of peaks."""
        return self._number_of_peaks

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset."""

    @property
    def parameters_hash(self) -> str:
        """Hash of the parameters."""
        return sha256(
            {
                "number_of_peaks": self._number_of_peaks,
                "fingerprints": [
                    fingerprint.__class__.__name__ for fingerprint in self._fingerprints
                ],
            }
        )

    def _load_preprocessed(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Load the preprocessed dataset."""
        padded_spectra = np.load(
            os.path.join(self.preprocessed_directory, "padded_spectra.npy")
        )
        smiles_ids = np.load(
            os.path.join(self.preprocessed_directory, "smiles_ids.npy")
        )

        spectral_features = np.load(
            os.path.join(self.preprocessed_directory, "spectral_features.npy")
        )

        fingerprints = {
            fingerprint.__class__.__name__: np.load(
                os.path.join(
                    self.preprocessed_directory, f"{fingerprint.__class__.__name__}.npy"
                )
            )
            for fingerprint in self._fingerprints
        }
        return padded_spectra, smiles_ids, spectral_features, fingerprints

    def _preprocess(self) -> None:
        """Preprocess the dataset."""
        spectra: List[Spectrum] = []
        for spectrum in tqdm(
            self.load(),
            desc=f"Loading {self.name}",
            disable=not self._verbose,
            leave=False,
            dynamic_ncols=True,
        ):
            spectrum = spectrum_processing(spectrum, self._number_of_peaks)
            if spectrum is not None:
                spectra.append(spectrum)

        # We identify the unique smiles in the dataset
        smiles_map: Dict[str, int] = {}
        smiles_ids: np.ndarray = np.zeros(len(spectra), dtype=np.int32)
        unique_smiles: List[str] = []
        for i, spectrum in enumerate(
            tqdm(
                spectra,
                desc="Extracting SMILES",
                disable=not self._verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            smile_id = smiles_map.get(spectrum.metadata.get("smiles"))
            if smile_id is None:
                smile_id = len(smiles_map)
                unique_smiles.append(spectrum.metadata.get("smiles"))
                smiles_map[spectrum.metadata["smiles"]] = len(smiles_map)
            smiles_ids[i] = smile_id

        fingerprints = {
            fingerprint.__class__.__name__: fingerprint.transform(unique_smiles)
            for fingerprint in tqdm(
                self._fingerprints,
                desc="Computing fingerprints",
                disable=not self._verbose or len(self._fingerprints) < 2,
                leave=False,
                dynamic_ncols=True,
            )
        }

        # We pad the spectra

        padded_spectra: np.ndarray = np.zeros(
            (len(spectra), self._number_of_peaks, 2), dtype=np.float32
        )

        for i, spectrum in enumerate(
            tqdm(
                spectra,
                desc="Padding spectra",
                disable=not self._verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            padded_spectra[i, : len(spectrum.peaks.mz), 0] = spectrum.peaks.mz / 2000.0
            padded_spectra[i, : len(spectrum.peaks.mz), 1] = spectrum.peaks.intensities

        # We compute the additional Spectral features
        spectral_features: np.ndarray = np.zeros((len(spectra), 3), dtype=np.float32)

        for i, spectrum in enumerate(
            tqdm(
                spectra,
                desc="Extracting spectral features",
                disable=not self._verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            spectral_features[i] = extract_mz_features(spectrum)

        # We store all the data

        os.makedirs(self.preprocessed_directory, exist_ok=True)

        np.save(
            os.path.join(self.preprocessed_directory, "padded_spectra.npy"),
            padded_spectra,
        )

        np.save(
            os.path.join(self.preprocessed_directory, "spectral_features.npy"),
            spectral_features,
        )

        np.save(
            os.path.join(self.preprocessed_directory, "smiles_ids.npy"),
            smiles_ids,
        )

        for fingerprint_name, fingerprint in fingerprints.items():
            np.save(
                os.path.join(self.preprocessed_directory, f"{fingerprint_name}.npy"),
                fingerprint,
            )

    @abstractmethod
    def load(self) -> Iterable[Spectrum]:
        """Load the dataset."""

    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Input shapes of the dataset."""
        return {
            "left_spectral_input": (self._number_of_peaks, 2),
            "left_spectral_features": (3,),
            "right_spectral_input": (self._number_of_peaks, 2),
            "right_spectral_features": (3,),
        }

    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Output shapes of the dataset."""
        return {
            fingerprint.__class__.__name__: (1,) for fingerprint in self._fingerprints
        }

    @property
    def number_of_outputs(self) -> int:
        """Number of outputs."""
        return len(self._fingerprints)

    def spectrum_processing(self, spectrum: Spectrum) -> Optional[Spectrum]:
        """This is how one would typically design a desired pre- and post-
        processing pipeline."""
        spectrum = default_filters(spectrum)
        spectrum = add_parent_mass(spectrum)
        spectrum = select_by_intensity(spectrum, intensity_from=0.05)
        spectrum = normalize_intensities(spectrum)
        spectrum = reduce_to_number_of_peaks(
            spectrum, n_required=5, ratio_desired=0.5, n_max=self._number_of_peaks
        )
        spectrum = select_by_mz(spectrum, mz_from=0, mz_to=2000)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum

    def __len__(self):
        # Number of batches per epoch
        return 10
        # int(
        #     self._indices.size * (self._indices.size - 1) / (2 * self._batch_size)
        # )

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # Generate indices for this batch
        indices = self._indices[
            index * self._batch_size : (index + 1) * self._batch_size
        ]

        left_spectral_batch: np.ndarray = np.zeros(
            (len(indices), self._number_of_peaks, 2), dtype=np.float32
        )
        left_spectral_features: np.ndarray = np.zeros(
            (len(indices), 3), dtype=np.float32
        )
        right_spectral_batch: np.ndarray = np.zeros(
            (len(indices), self._number_of_peaks, 2), dtype=np.float32
        )
        right_spectral_features: np.ndarray = np.zeros(
            (len(indices), 3), dtype=np.float32
        )
        similarities = np.zeros(
            (len(indices), self.number_of_outputs), dtype=np.float32
        )

        for i, (idx1, idx2) in enumerate(
            np.random.choice(self._indices, (len(indices), 2), replace=False)
        ):
            left_spectral_batch[i] = self._padded_spectra[idx1]
            left_spectral_features[i] = self._spectral_features[idx1]
            right_spectral_batch[i] = self._padded_spectra[idx2]
            right_spectral_features[i] = self._spectral_features[idx2]

            left_smile_id = self._smiles_ids[idx1]
            right_smile_id = self._smiles_ids[idx2]

            for j, fingerprint in enumerate(self._fingerprints_data.values()):
                similarities[i, j] = jaccard_similarity(
                    fingerprint[left_smile_id], fingerprint[right_smile_id]
                )

        assert np.isfinite(similarities).all()
        assert np.isfinite(left_spectral_batch).all()
        assert np.isfinite(left_spectral_features).all()
        assert np.isfinite(right_spectral_batch).all()
        assert np.isfinite(right_spectral_features).all()

        return (
            {
                "left_spectral_input": left_spectral_batch,
                "left_spectral_features": left_spectral_features,
                "right_spectral_input": right_spectral_batch,
                "right_spectral_features": right_spectral_features,
            },
            similarities,
        )

    def on_epoch_end(self):
        np.random.shuffle(self._indices)
