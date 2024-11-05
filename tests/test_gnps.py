"""Test suite for the GNPS dataset."""

from multiprocessing import cpu_count
from skfp.fingerprints import LayeredFingerprint
from spectromer.datasets import GNPS


def test_gnps():
    """Test the GNPS dataset."""
    gnps = GNPS(
        fingerprints=[LayeredFingerprint(n_jobs=cpu_count(), verbose=False)],
    )

    # assert len(gnps) == 3582598579

    print(gnps[0])