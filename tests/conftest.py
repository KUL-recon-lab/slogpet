"""Shared fixtures.

The tests import ``slogpet`` from the repository, so they run without the
package being installed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slogpet as sp                                    # noqa: E402


@pytest.fixture(scope="session")
def systems():
    from slogpet.data import load_systems
    return load_systems()


@pytest.fixture(scope="session")
def groups():
    from slogpet.data import load_detector_groups
    return load_detector_groups()


@pytest.fixture(scope="session")
def quadra():
    """A concrete, published system to exercise the assembly with."""
    return sp.Scanner("Quadra", L_pet=1060.0, D_pet=820.0, F_y=3.4, F_z=3.8,
                      ctr=230.0, S_nema=175.3)


@pytest.fixture(scope="session")
def task():
    return sp.Task(F_o=5.0, D_cyl=300.0)
