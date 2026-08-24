import torch

from scripts.run_owt_dct_band_ablation import equal_power_subsets, octave_bands


def test_octave_bands_cover_every_mode_once():
    bands = octave_bands(1024)
    assert bands == [
        (0, 1),
        (1, 3),
        (3, 5),
        (5, 9),
        (9, 17),
        (17, 33),
        (33, 65),
        (65, 129),
        (129, 257),
        (257, 513),
        (513, 1024),
    ]
    assert [mode for start, stop in bands for mode in range(start, stop)] == list(
        range(1024)
    )


def test_equal_power_controls_are_close():
    power = torch.ones(16)
    subsets = equal_power_subsets(power, (3, 7), repeats=3, seed=4)
    for subset in subsets:
        assert len(subset) == 4


def test_zero_power_control_can_remove_nothing():
    power = torch.ones(8)
    power[0] = 0
    [subset] = equal_power_subsets(power, (0, 1), repeats=1, seed=0)
    assert len(subset) == 0
