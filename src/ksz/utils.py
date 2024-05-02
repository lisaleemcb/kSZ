"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = ksz.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
import numpy as np
import copy as cp

from scipy.interpolate import CubicSpline
from catwoman.utils import find_index
from ksz.parameters import modelparams_Gorce2022
from ksz import __version__

__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from ksz.skeleton import fib`,
# when using this Python module as a library.

def dimless(k, P):
    return (k**3.0 * P) / (2 * np.pi**2)

def tau(n):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        tau (array_like): tau (cumulative integral)
    """
    z_unity = 5.0 # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra)

    spl = CubicSpline(x, y)

    return z, tau

def xe_allz(z, xe):
    """Calculates tau given an ionisation history xe(z)

    Args:
        z  (array_like): redshift
        xe (array_like): ionisation history

    Returns:
        z  (array_like): redshift
        xe (array_like): extrapolated up to z=0
    """
    xe_recomb = 1.7e-4
    Yp = 0.2453
    not4 = 3.9715 #eta
    fHe = Yp/(not4*(1-Yp))

    z_unity = 5.0 # redshift at which hydrogen is has an ionisation fraction x_HII=1
    z_HeII = 3.5  # redshift at which helium doubly ionises

    if z.min() <= z_unity:
        z_extra = np.linspace(0.0, z.min(), endpoint=False)
    else:
        z_extra = np.linspace(0.0, z_unity)

    xe_lowz = np.ones_like(z_extra) * 1.08 + xe_recomb

    z2interpl = np.concatenate((z_extra, np.sort(z)))
    xe2interpl = np.concatenate((xe_lowz, xe[::-1]))

    xe_all = CubicSpline(z2interpl, xe2interpl)

    z_all = np.linspace(0.0, z.max(), 1000)
    add_He = (z_all < 3.5).astype(float) * fHe

    return z_all, np.minimum(xe_all(z_all), (1.08 + xe_recomb)) + add_He

def unpack_data(sim, zrange, krange):

    ksize = krange[1]-krange[0]
    if isinstance(zrange, int):
        data = sim.Pee[zrange]['P_k'][krange[0]:krange[1]]
    else:
        zsize = zrange[1]-zrange[0]
        data = np.zeros((zsize, ksize))
        for i, zi in enumerate(range(zrange[0], zrange[1])):
            data[i] = sim.Pee[zi]['P_k'][krange[0]:krange[1]]

    return data

def pack_params(params):

    model_params = cp.deepcopy(modelparams_Gorce2022)
    for i in range(len(params)):
        p = params[i]
        if i == 0:
            p = 10.**p

        key = list(model_params.keys())[i]
        model_params[key] = p

    return model_params
