import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trust import calibration as CAL
from src.xai.methods import energy_batch
from src.classification.evaluate import wilson_ci


def test_ece_is_zero_when_confidence_equals_accuracy():
    conf = np.full(100, 0.5); correct = np.array([1] * 50 + [0] * 50)
    assert CAL.ece_mce(conf, correct)[0] < 1e-9


def test_temperature_scaling_recovers_known_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (20000, 4)); p = CAL.softmax_T(z)
    y = (p.cumsum(1) > rng.random((20000, 1))).argmax(1)           # labels drawn from the model itself => perfectly calibrated
    T = CAL.fit_temperature(3.0 * z, y)                              # logits inflated 3x => the fit must undo it
    assert 2.6 < T < 3.4


def test_energy_inside_half_mask_is_half():
    cam = np.ones((1, 8, 8)); mask = np.zeros((1, 8, 8), bool); mask[:, :, :4] = True
    assert abs(energy_batch(cam, mask)[0] - 0.5) < 1e-9


def test_wilson_interval_contains_proportion():
    lo, hi = wilson_ci(45, 50); assert lo < 0.9 < hi
