# Spectromer

A pretrained spectra transformer.

## Installation

As usual, you can install Spectromer using pip:

ACTUALLY IT IS NOT YET PUBLISHED ON PYPI, WILL DO SOON

```bash
pip install spectromer
```

## Training Spectromer

To train Spectromer, you can use the `spectromer train` command. The following command trains Spectromer on the GNPS dataset with 256 peaks, 256 embedding size, 1000 epochs, 4096 batch size, and saves the model to `v1.keras`.

```bash
spectromer train --dataset GNPS --number-of-peaks 256 --embedding-size 512 --epochs 10000 --batch-size 1024 --output v1.keras --verbose
```

## Embedding spectra

To embed spectra, you can use the `spectromer embed` command. The following command embeds the spectra in `spectra.mgf` using the model `v1.keras` and saves the embeddings to `embeddings.npy`.

```bash
spectromer transform --input datasets/GNPS/matchms.mgf --model v1.keras --output matchms.npy --verbose
```
