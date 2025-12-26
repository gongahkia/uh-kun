from pathlib import Path

import pytest

from uh_kun.labels import infer_label_from_path, normalize_label


def test_normalize_label_aliases():
    assert normalize_label("ya") == "Ya-kun"
    assert normalize_label("YaKun") == "Ya-kun"
    assert normalize_label("no-kun") == "No-kun"
    assert normalize_label("maybe") == "Maybe-kun"


def test_infer_label_from_path():
    p = Path("/tmp/data/train/ya-kun/img.jpg")
    assert infer_label_from_path(p) == "Ya-kun"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_label("unknown")
