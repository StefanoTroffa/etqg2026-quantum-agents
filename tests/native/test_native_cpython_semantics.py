"""The native engine reproduces two CPython behaviours the Python simulator's numbers depend on: float `sum()` (exact
compensated summation since 3.12: cheqqers_official.py:2162, expecti_cheqqers.py:198, cheqqers_official.py:2780) and
the iteration order of a set of small ints (the `victims` sets summed in eval_features, cheqqers_official.py:2780).
Checked here against the running interpreter on random inputs."""

import ctypes
import random

import pytest

from search import native_cheqqers as nc

if not nc.available():
    pytest.skip("native Cheqqers library unavailable", allow_module_level=True)

LIB = nc.load()
LIB.cq_test_set_order.restype = ctypes.c_int
LIB.cq_test_set_order.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
LIB.cq_test_sum.restype = ctypes.c_double
LIB.cq_test_sum.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]


@pytest.mark.parametrize("universe", [18, 32, 50])
def test_set_iteration_order_is_cpythons(universe):
    rng = random.Random(universe)
    for _ in range(3000):
        keys = [rng.randrange(universe) for _ in range(rng.randrange(1, 40))]
        out = (ctypes.c_int32 * 64)()
        n = LIB.cq_test_set_order((ctypes.c_int32 * len(keys))(*keys), len(keys), out)
        expected = set()
        for k in keys:
            expected.add(k)
        assert list(out[:n]) == list(expected), keys


def test_float_sum_is_cpythons():
    rng = random.Random(7)
    for _ in range(20000):
        values = [rng.choice([rng.random(), rng.random() * 1e-9, 1.0 - rng.random() * 1e-12, rng.uniform(-1, 1)])
                  for _ in range(rng.randrange(0, 9))]
        got = LIB.cq_test_sum((ctypes.c_double * max(1, len(values)))(*values), len(values))
        assert nc_bits(got) == nc_bits(float(sum(values))), values


def nc_bits(x: float) -> bytes:
    import struct
    return struct.pack("d", x)
