from __future__ import annotations

import fast_viterbi as m


def test_version():
    assert m.__version__ == "0.1.1"


def test_add():
    assert m.add(1, 2) == 3


def test_sub():
    assert m.subtract(1, 2) == -1
