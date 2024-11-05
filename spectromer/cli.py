"""CLI for Spectromer."""

from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
import os
from skfp.fingerprints import (
    LayeredFingerprint,
    ECFPFingerprint,
    TopologicalTorsionFingerprint,
)
import pandas as pd
import numpy as np
from keras.api.callbacks import History
from keras.api.backend import clear_session

from spectromer.datasets import AVAILABLE_DATASETS, Dataset
from spectromer.model import Spectromer


def train(args: Namespace) -> None:
    """Train the Spectromer model."""
    clear_session()
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
    except ImportError:
        pass

    dataset: Dataset = AVAILABLE_DATASETS[args.dataset](
        fingerprints=[
            LayeredFingerprint(n_jobs=cpu_count(), verbose=0),
            ECFPFingerprint(n_jobs=cpu_count(), verbose=0),
            TopologicalTorsionFingerprint(n_jobs=cpu_count(), verbose=0),
        ],
        batch_size=args.batch_size,
        number_of_peaks=args.number_of_peaks,
        directory=args.directory,
        verbose=args.verbose,
    )

    model = Spectromer(
        path=(
            "checkpoint.keras"
            if args.resume and os.path.exists("checkpoint.keras")
            else None
        )
    )
    history: History = model.fit(
        dataset, embedding_size=args.embedding_size, epochs=args.epochs
    )
    model.save(args.output)
    pd.DataFrame(history.history).to_csv(args.history_path, index=False)


def build_train_parser(parser: ArgumentParser) -> None:
    """Build the parser for the train command."""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS.keys(),
        help="The dataset to train on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="The batch size.",
    )
    parser.add_argument(
        "--number-of-peaks",
        type=int,
        default=512,
        help="The number of peaks.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="datasets",
        help="The directory to store the dataset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes training from checkpoint.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        required=True,
        help="The embedding size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="The number of epochs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output path for the model.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="history.csv",
        help="The output path for the history.",
    )

    parser.set_defaults(func=train)


def transform(args: Namespace) -> None:
    """Transform the input data."""
    model = Spectromer(path=args.model)
    embedding: np.ndarray = model.transform(spectra=args.input, verbose=args.verbose)
    np.save(args.output, embedding)


def build_transform_parser(parser: ArgumentParser) -> None:
    """Build the parser for the transform command."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the model.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to the spectra.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output path for the embedding.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    parser.set_defaults(func=transform)


def main() -> None:
    """Main entry point."""
    parser = ArgumentParser(description="CLI for Spectromer.")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="Train the Spectromer model.")
    build_train_parser(train_parser)

    transform_parser = subparsers.add_parser("transform", help="Transform the input data.")
    build_transform_parser(transform_parser)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
