import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import ksz.utils as utils
from ksz.parameters import *
import ksz.Pee
import ksz.analyse as analyse
import ksz.KSZ

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat


# Pee_path = '/Users/emcbride/spectra/Pee'
# kSZ_path = '/Users/emcbride/spectra/kSZ'
# fits_path = '/Users/emcbride/lklhd_files'
# params_path = '/Users/emcbride/kSZ/data/LoreLi_summaries/param_files'

Pee_path = '/home/emc-brid/spectra/Pee'
kSZ_path = '/home/emc-brid/spectra/kSZ'
fits_path = '/home/emc-brid/lklhd_files'
params_path = '/home/emc-brid/param_files'

pattern = re.compile(r"simu(\d+)\_Pee_spectra.npz")

# List to store the extracted numbers
sims = []
# Loop through files in the directory
for filename in os.listdir(Pee_path):
    match = pattern.match(filename)
    if match:
        # Extract the number (as an integer) and store it
        sims.append(int(match.group(1)))

print(f'There are {len(sims)} sims available to parse! Getting started...')
print()

ells = np.linspace(1,15000, 100)

for sn in sims:
    print('==================================')
    print(f'Now on sim {sn}')
    print('==================================')

    fit_fn = f'{fits_path}/bestfit_params_simu{sn}.npz'
    Gorce_fn = f'{kSZ_path}/Gorce/kSZ_Gorce_simu{sn}'
    LoReLi_fn = f'{kSZ_path}/LoReLi/kSZ_LoReLi_simu{sn}'

    print()
    print('loading params...')
    if os.path.exists(f'{Gorce_fn}.npz'):
        if os.path.isfile(f'{Gorce_fn}.npz'):
            print('Spectra already calculated, skipping...')
            continue

    if os.path.exists(fit_fn):
        if os.path.isfile(fit_fn):
           bf = np.load(fit_fn, allow_pickle=True) 
        else:
            print('no fits file! skipping...')
            continue
    print(f'params for sim {sn} loaded...')
    bf = bf['bf'].item()
    print(bf)
    print()
    alpha0 = bf[str(sn)]['alpha0']
    kappa = bf[str(sn)]['kappa']

    print('loading data...')
    #data = np.load(f'{Pee_path}/simu{sn}_Pee_spectra.npz', allow_pickle=True)
    sim = Cat(sn, path_spectra=Pee_path, path_params=params_path, verbose=True)
    print('data loaded...')

    print('')

    print('smoothing Pee...')
    k = []
    Pee = []
    for i in range(0, sim.k.size - 1,2):
        k.append((sim.k[i] + sim.k[i + 1]) / 2.0)
        Pee.append((sim.Pee[:,i] +sim.Pee[:,i + 1]) / 2.0)

    Pee = np.asarray(Pee).T

    print('Pee smoothed...')
    print()

    print('simulating Gorce spectrum...')
    print('----------------------------')

    #sim = Cat(sn, verbose=True)
    Gorce = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=False,
                Pee_data=None, xe_data=sim.xe, z_data=sim.z, k_data=None, alpha0=alpha0, kappa=kappa,
                kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
    
    print()
    print('simulating LoReLi spectrum...')
    print('----------------------------')
    
    # LoReLi = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
    #             Pee_data=sim.Pee, xe_data=sim.xe, z_data=sim.z, k_data=sim.k, alpha0=alpha0, kappa=kappa,
    #             kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
    
    LoReLi_smoothed = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
                Pee_data=Pee, xe_data=sim.xe, z_data=sim.z, k_data=k, alpha0=alpha0, kappa=kappa,
                kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
    
    print()
    
    np.savez(Gorce_fn, ells=ells, kSZ=Gorce)
    np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi_smoothed)
    #np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi)

    print(f'saving spectra for simulation {sn}...')

