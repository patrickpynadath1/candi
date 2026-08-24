import numpy as np
import torch

from scripts.run_blur_basis_study import feature_spectra, spectrum_metrics


def test_spectrum_metrics_uniform_effective_dimension():
    metrics = spectrum_metrics(np.ones(100))
    assert abs(metrics["participation_ratio"] - 100) < 1e-10
    assert abs(metrics["top_10pct_power"] - 0.1) < 1e-10
    assert metrics["modes_to_50pct"] == 50


def test_feature_spectra_are_descending_and_norm_corrected():
    weight = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    spectra, correlations = feature_spectra(weight, torch.ones(4), 0, 0.02)
    assert spectra["vocab_centered"][0] >= spectra["vocab_centered"][1]
    assert len(correlations) == 2
    np.testing.assert_allclose(
        (spectra["frequency_weighted"] / spectra["frequency_weighted"].sum()).sum(),
        1.0,
    )
