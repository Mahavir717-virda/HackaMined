import math

from scripts.evaluate_model import normalize_label


def test_normalize_label_handles_common_values():
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0
    assert normalize_label("Critical") == 1
    assert normalize_label("Low Risk") == 0
    assert normalize_label("Clear") == 0
    assert normalize_label("Flagged") == 1
    assert math.isnan(normalize_label("UNKNOWN_LABEL"))
