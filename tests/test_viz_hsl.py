import numpy as np

from treelidarnet.viz.hsl import combo_colors, combo_from_arrays, hsl_to_rgb


def test_hsl_to_rgb_red_hue_zero():
    r, g, b = hsl_to_rgb(0, 0.38, 0.52)
    assert r >= g and r >= b
    assert 0 <= r <= 255


def test_combo_colors_hue_rises_with_score():
    v2 = np.array([0.0, 0.5, 1.0])
    v3 = np.array([0.0, 0.5, 1.0])
    rgb = combo_colors(v2, v3, hue_low=0.0, hue_high=142.0, sat=0.38, light=0.52)
    assert rgb.shape == (3, 3)
    assert rgb[0, 0] >= rgb[2, 0]  # low combo → red hue; high → green (142°)
    assert rgb[2, 1] >= rgb[0, 1]


def test_combo_from_arrays_weighted():
    c = combo_from_arrays(np.array([0.6]), np.array([0.2]), w_v2=0.5, w_v3=0.5)
    assert float(c[0]) == 0.4