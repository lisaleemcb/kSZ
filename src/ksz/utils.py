import argparse
import logging
import sys
import numpy as np
import copy as cp

from scipy.interpolate import CubicSpline
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

def unpack_data(spectra_dict):
    data = np.zeros((len(spectra_dict), spectra_dict[0]['P_k'].size))

    # if isinstance(zrange, int):
    #     data = spectra[zrange][key][krange[0]:krange[1]]
    for i in range(len(spectra_dict)):
        data[i] = spectra_dict[i]['P_k']

    return data

def pack_params(pvals, pfit):
    params = cp.deepcopy(modelparams_Gorce2022)
    for i, key in enumerate(pfit):
        params[key] = pvals[i]

    return params

def unpack_params(params, pfit):
    return np.asarray([params[key] for key in pfit])

def find_index(arr):
    for i in range(arr.size - 1):
        a = arr[i:]
       # print(f'array looks like {a}')
        if np.all(a[:-1] < a[1:]):
            return i

    print('No monotonically increasing part of this function. Are you sure this is correct?')
    return NaN

# import matplotlib as m
# cmap = m.cm.get_cmap('Blues')
# norm = m.colors.Normalize(vmin=min_chi2-10, vmax=min_chi2+20.)
# lvs = [min_chi2+2.30,min_chi2+6.17,min_chi2+11.8]
# labels=[r'$68\%$',r'$95\%$',r'$99.7\%$']

# plt.figure()
# CS = plt.contour(kappas,alphas,chi2,levels=lvs,colors=[cmap(norm(lvs[0])),cmap(norm(lvs[1])),cmap(norm(lvs[2]))])#,linewidths=.8,colors='white')
# plt.xlabel(r'$\kappa$ [Mpc$^{-1}$]',fontsize=13)
# # plt.xlim(0.07,0.085)
# plt.ylabel(r'log$\alpha_0$',fontsize=13)
# # plt.ylim(4.05,4.2)
# ax = plt.gca()
# fmt={}
# for l,s in zip(lvs, labels):
#     fmt[l]=s
# ax.clabel(CS,CS.levels,fmt=fmt,inline=True)
# plt.scatter(kappa,alpha0,color='k',marker='+',s=100,lw=1.)

# kappas_sims = np.array([0.093,0.094,0.098,0.100,0.089,0.093])
# alphas_sims = np.array([3.86,3.85,3.80,3.78,3.91,3.87])
# ax.scatter(kappas_sims,alphas_sims,color='navy',label='Our simulations',marker='+', s=80,zorder=10)

# plt.tight_layout()
