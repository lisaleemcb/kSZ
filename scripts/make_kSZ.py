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
# params_path = '/Users/emcbride/lklhd_files'

Pee_path = '/home/emc-brid/spectra/Pee'
kSZ_path = '/home/emc-brid/spectra/kSZ'
params_path = '/home/emc-brid/lklhd_files'


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

    param_fn = f'{params_path}/bestfit_params_simu{sn}.npz'
    Gorce_fn = f'{kSZ_path}/Gorce/kSZ_Gorce_simu{sn}'
    LoReLi_fn = f'{kSZ_path}/LoReLi/kSZ_LoReLi_simu{sn}'

    if os.path.exists(Gorce_fn):
        if os.path.isfile(Gorce_fn):
            print('Spectra already calculated, skipping...')
            continue

    if os.path.exists(param_fn):
        if os.path.isfile(param_fn):
            bf = np.load(param_fn, allow_pickle=True)
        else:
            continue

    bf = bf['bf'].item()
    print(bf)
    print()
    alpha0 = bf[str(sn)]['alpha0']
    kappa = bf[str(sn)]['kappa']

    data = np.load(f'{Pee_path}/simu{sn}_Pee_spectra.npz', allow_pickle=True)

    k = []
    Pee = []
    for i in range(0,data['k'].size - 1,2):
        k.append((data['k'][i] + data['k'][i + 1]) / 2.0)
        Pee.append((data['Pk'][:,i] + data['Pk'][:,i + 1]) / 2.0)

    Pee = np.asarray(Pee).T

    #sim = Cat(sn, verbose=True)
    Gorce = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=False,
                Pee_data=None, xe_data=data['xe'], z_data=data['z'], k_data=None, alpha0=alpha0, kappa=kappa,
                kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
    
    print()
    
    LoReLi = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
                Pee_data=data['Pk'], xe_data=data['xe'], z_data=data['z'], k_data=data['k'], alpha0=alpha0, kappa=kappa,
                kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
    
    print()
    
    np.savez(Gorce_fn, ells=ells, kSZ=Gorce)
    np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi)

    print(f'saving spectra for simulation {sn}...')

