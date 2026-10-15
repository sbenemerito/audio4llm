"""Key estimate (audio4llm/key.py): pure profile matching on synthetic chroma, no audio models."""

import numpy as np
import pytest

from audio4llm.key import KK_MAJOR, KK_MINOR, PITCH_CLASSES, estimate_key, harmonic_chroma, relative_key

C_MAJOR_SCALE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)


def triad(root: int, minor: bool = False) -> np.ndarray:
    c = np.zeros(12)
    c[[root % 12, (root + (3 if minor else 4)) % 12, (root + 7) % 12]] = 1
    return c


def test_c_major_scale_is_c_major_with_relative_runner_up():
    k = estimate_key(C_MAJOR_SCALE)
    assert (k["label"], k["tonic"], k["mode"]) == ("C major", "C", "major")
    assert k["runner_up"] == "A minor" == k["relative"]
    assert k["correlation"] > k["runner_up_correlation"]


@pytest.mark.parametrize("tonic", range(12))
def test_rotated_profiles_give_the_right_tonic(tonic):
    major = estimate_key(np.roll(KK_MAJOR, tonic))
    minor = estimate_key(np.roll(KK_MINOR, tonic))
    assert major["label"] == f"{PITCH_CLASSES[tonic]} major"
    assert minor["label"] == f"{PITCH_CLASSES[tonic]} minor"
    assert major["correlation"] == pytest.approx(1.0) and major["confidence"] == 1.0


@pytest.mark.parametrize("tonic", range(12))
def test_rotated_scale_gives_the_right_tonic(tonic):
    assert estimate_key(np.roll(C_MAJOR_SCALE, tonic))["tonic"] == PITCH_CLASSES[tonic]


def test_minor_tonic_triad_weighting_gives_minor():
    # Am F C G with the tonic chord twice as long: A minor; the same loop evenly weighted leans to C major
    loop = triad(9, minor=True) + triad(5) + triad(0) + triad(7)
    assert estimate_key(loop + triad(9, minor=True))["label"] == "A minor"
    assert estimate_key(loop)["label"] in ("C major", "A minor")


def test_relative_keys():
    assert relative_key(0, "major") == (9, "minor")
    assert relative_key(9, "minor") == (0, "major")
    assert estimate_key(np.roll(KK_MINOR, 4))["relative"] == "G major"  # E minor


def test_flat_or_silent_chroma_has_no_key():
    for c in (np.zeros(12), np.ones(12)):
        k = estimate_key(c)
        assert k["label"] is None and k["confidence"] == 0.0
        assert "estimate" in k["method"]


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        estimate_key(np.ones(11))


def test_confidence_is_low_for_random_chroma_and_bounded():
    rng = np.random.default_rng(0)
    conf = [estimate_key(rng.dirichlet(np.ones(12)))["confidence"] for _ in range(300)]
    assert all(0.0 <= c <= 1.0 for c in conf)
    assert np.median(conf) < 0.15
    assert estimate_key(C_MAJOR_SCALE + 0.5 * triad(0))["confidence"] > np.percentile(conf, 90)


def test_method_says_estimate():
    assert estimate_key(C_MAJOR_SCALE)["method"].startswith("estimate")


def test_harmonic_chroma_of_synthetic_a_minor_chord():
    sr = 22050
    t = np.arange(int(3.5 * sr)) / sr  # CQT needs ~3 s for its lowest octave
    y = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) for m in (57, 60, 64))  # A3 C4 E4
    c = harmonic_chroma(y.astype(np.float32), sr)
    assert c.shape == (12,) and c.sum() == pytest.approx(1.0)
    assert set(np.argsort(c)[-3:]) == {9, 0, 4}  # A, C, E
    assert estimate_key(c)["label"] in ("A minor", "C major")
