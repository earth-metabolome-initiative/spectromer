"""Spetromer embedding model."""

from typing import Optional, Tuple, Dict, Union, Iterable, List
from keras.api.models import Model
from keras.api.callbacks import History
from keras.api.layers import (
    Input,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Multiply,
    Dropout,
    Subtract,
    BatchNormalization,
    Concatenate,
)
from keras.api.initializers import HeNormal
from keras.api.utils import plot_model
from keras.api.saving import load_model
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm
from matchms import Spectrum
from matchms.importing import (
    load_from_mgf,
    load_from_msp,
    load_from_mzml,
    load_from_mzxml,
)
import numpy as np
from spectromer.datasets.dataset import (
    Dataset,
    spectrum_processing,
    extract_mz_features,
)


class Spectromer:
    """Spectromer model."""

    def __init__(self, path: Optional[str] = None) -> None:
        """Initialize the Spectromer model."""
        self._model: Optional[Model] = load_model(path) if path else None

    @property
    def number_of_peaks(self) -> int:
        """Return the number of peaks."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        return self._model.input_shape[0][1]

    @property
    def embedding_size(self) -> int:
        """Return the embedding size."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        return self._model.output_shape[1]

    def siamese(self, number_of_peaks: int, embedding_size: int) -> Model:
        """Build the Siamese Spectromer model."""
        padded_spectra_input = Input(shape=(number_of_peaks, 2), name="padded_spectra")
        spectral_features_input = Input(shape=(3,), name="spectral_features")

        spectral_features_size: int = embedding_size // 6

        spectral_features = spectral_features_input
        for _ in range(3):
            spectral_features = Dense(
                spectral_features_size,
                activation="relu",
                kernel_initializer=HeNormal(),
            )(spectral_features)
            spectral_features = BatchNormalization()(spectral_features)
            spectral_features = Dropout(0.3)(spectral_features)

        reduced_embedding_size = embedding_size - spectral_features_size

        padded_spectra = padded_spectra_input
        for _ in range(5):
            padded_spectra = Conv1D(
                filters=reduced_embedding_size,
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_initializer=HeNormal(),
            )(padded_spectra)
            padded_spectra = BatchNormalization()(padded_spectra)
            spectral_features = Dropout(0.3)(spectral_features)

        # for _ in range(2):
        #     padded_spectra = TransformerEncoder(
        #         intermediate_dim=reduced_embedding_size,
        #         num_heads=2,
        #         dropout_rate=0.3,
        #         kernel_initializer=HeNormal(),
        #     )(padded_spectra)

        padded_spectra = GlobalAveragePooling1D()(padded_spectra)

        concatenated = Concatenate()([padded_spectra, spectral_features])

        output = Dense(
            embedding_size,
            activation="linear",
            kernel_initializer=HeNormal(),
        )(concatenated)

        return Model(
            inputs={
                "padded_spectra": padded_spectra_input,
                "spectral_features": spectral_features_input,
            },
            outputs=output,
            name="Spectromer",
        )

    def build(
        self,
        number_of_peaks: int,
        embedding_size: int,
        input_shapes: Dict[str, Tuple[int, ...]],
        number_of_outputs: int,
    ) -> Tuple[Model, Model]:
        """Compile the Spectromer model."""
        siamese: Model = self.siamese(number_of_peaks, embedding_size)

        left_spectral_input = Input(
            shape=input_shapes["left_spectral_input"], name="left_spectral_input"
        )
        right_spectral_input = Input(
            shape=input_shapes["right_spectral_input"], name="right_spectral_input"
        )
        left_spectral_features = Input(
            shape=input_shapes["left_spectral_features"], name="left_spectral_features"
        )
        right_spectral_features = Input(
            shape=input_shapes["right_spectral_features"],
            name="right_spectral_features",
        )

        left_output = siamese(
            {
                "padded_spectra": left_spectral_input,
                "spectral_features": left_spectral_features,
            }
        )
        right_output = siamese(
            {
                "padded_spectra": right_spectral_input,
                "spectral_features": right_spectral_features,
            }
        )

        subtraction = Subtract()([left_output, right_output])
        squared = Multiply()([subtraction, subtraction])

        output = Dense(
            units=number_of_outputs,
            activation="linear",
            kernel_initializer=HeNormal(),
        )(squared)

        model = Model(
            inputs={
                "left_spectral_input": left_spectral_input,
                "right_spectral_input": right_spectral_input,
                "left_spectral_features": left_spectral_features,
                "right_spectral_features": right_spectral_features,
            },
            outputs=output,
        )

        model.compile(
            loss="mse",
            optimizer="adam",
            jit_compile=True,
        )

        return siamese, model

    def fit(
        self,
        dataset: Dataset,
        embedding_size: int,
        epochs: int = 1000,
    ) -> History:
        """Fit the Spectromer model."""

        train, valid = dataset.split()
        siamese, model = self.build(
            number_of_peaks=train.number_of_peaks,
            embedding_size=embedding_size,
            input_shapes=train.input_shapes(),
            number_of_outputs=train.number_of_outputs,
        )

        plot_model(
            siamese, show_shapes=True, show_layer_names=True, to_file="siamese.png"
        )
        plot_model(model, show_shapes=True, show_layer_names=True, to_file="model.png")

        history = model.fit(
            train,
            validation_data=valid,
            steps_per_epoch=len(train),
            validation_steps=len(valid),
            epochs=epochs,
            verbose=0,
            callbacks=[
                TqdmCallback(
                    verbose=1,
                    leave=False,
                ),
                ModelCheckpoint(
                    filepath="checkpoint.keras",
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=100,
                    min_lr=1e-6,
                    verbose=0,
                ),
                # TerminateOnNaN(),
            ],
        )
        self._model = siamese

        return history

    def save(self, path: str) -> None:
        """Save the Spectromer model."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        self._model.save(path)

    def transform(
        self, spectra: Union[str, Iterable[Spectrum]], verbose: bool = True
    ) -> np.ndarray:
        """Embed spectra."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")

        if isinstance(spectra, str):
            if spectra.endswith(".msp"):
                spectra = load_from_msp(spectra)
            elif spectra.endswith(".mzml"):
                spectra = load_from_mzml(spectra)
            elif spectra.endswith(".mzxml"):
                spectra = load_from_mzxml(spectra)
            elif spectra.endswith(".mgf"):
                spectra = load_from_mgf(spectra)
            else:
                raise ValueError("Unknown format.")

        processed_spectra: List[Spectrum] = []
        for spectrum in tqdm(
            spectra,
            desc="Preprocessing spectra",
            disable=not verbose,
            leave=False,
            dynamic_ncols=True,
        ):
            processed_spectra.append(
                spectrum_processing(spectrum, number_of_peaks=self.number_of_peaks)
            )

        padded_spectra: np.ndarray = np.zeros(
            (len(processed_spectra), self.number_of_peaks, 2)
        )

        for i, spectrum in enumerate(
            tqdm(
                processed_spectra,
                desc="Padding spectra",
                disable=not verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            if spectrum is None:
                continue
            padded_spectra[i, : len(spectrum.peaks), 0] = spectrum.peaks.mz / 2000.0
            padded_spectra[i, : len(spectrum.peaks), 1] = spectrum.peaks.intensities

        spectral_features: np.ndarray = np.zeros((len(processed_spectra), 3))

        for i, spectrum in enumerate(
            tqdm(
                spectra,
                desc="Extracting spectral features",
                disable=not verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            spectral_features[i] = extract_mz_features(spectrum)

        return self._model.predict(
            {
                "padded_spectra": padded_spectra,
                "spectral_features": spectral_features,
            }
        )
