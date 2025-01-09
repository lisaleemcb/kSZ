import os
import re
import time
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


import argparse
import numpy as np

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a numpy file which is a list of sims.")
    parser.add_argument("--file", type=str, help="Path to the numpy file (.npy or .npz)")
    
    # Parse arguments
    args = parser.parse_args()

    # Pee_path = '/Users/emcbride/spectra/Pee'
    # kSZ_path = '/Users/emcbride/spectra/kSZ'
    # fits_path = '/Users/emcbride/lklhd_files'
    # params_path = '/Users/emcbride/kSZ/data/LoreLi_summaries/param_files'

    sim_path = 'home/emc-brid/ps_ee'
    ion_path = 'home/emc-brid/ion_histories_full.npz'
    Pee_path = '/home/emc-brid/spectra/Pee'
    kSZ_path = '/home/emc-brid/spectra/kSZ'
    fits_path = '/home/emc-brid/lklhd_files'
    params_path = '/home/emc-brid/param_files'

    sim_path = '/jet/home/emcbride/ps_ee'
    ion_path = '/jet/home/emcbride/ion_histories_full.npz'
    Pee_path = '/jet/home/emcbride/spectra/Pee'
    kSZ_path = '/jet/home/emcbride/LoReLi_format_kSZ'
    fits_path = '/jet/home/emcbride/lklhd_files'
    params_path = '/jet/home/emcbride/param_files'

    if args.file:
        if os.path.exists(args.file):
            process_file(args.file)
                # Load the numpy file
            file_path = args.file
            sims = np.load(file_path)
            print('Sims list loaded for this run:')
            print(f'\t {sims}')

    else:
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


    ells = np.linspace(1,15000, 30)

    print(f'Now simulating {len(sims)} kSZ spectra!')
    for j, sn in enumerate(sims):
        start_time = time.time()
        print('==================================')
        print(f'Now on the {j+1}th run for sim {sn}')
        print('==================================')

       # fit_fn = f'{fits_path}/bestfit_params_simu{sn}.npz'
    # Gorce_fn = f'{kSZ_path}/Gorce/kSZ_Gorce_simu{sn}'
        LoReLi_fn = f'{kSZ_path}/LoReLi/kSZ_LoReLi_simu{sn}'

        if os.path.exists(f'{LoReLi_fn}.npz'):
            if os.path.isfile(f'{LoReLi_fn}.npz'):
                print('Spectra already calculated, skipping...')
                continue


        print()
        print('loading params...')
        # if os.path.exists(fit_fn):
        #     if os.path.isfile(fit_fn):
        #         bf = np.load(fit_fn, allow_pickle=True) 
        #     print(fit_fn)
        # else:
        #     print('no fits file! skipping...')
        #     continue

        # print(f'params for sim {sn} loaded...')
        # bf = bf['bf'].item()
        # print(bf)
        # print()
        # alpha0 = bf[str(sn)]['alpha0']
        # kappa = bf[str(sn)]['kappa']

        print('loading data...')
        #data = np.load(f'{Pee_path}/simu{sn}_Pee_spectra.npz', allow_pickle=True)
        sim_check = Cat(sn, skip_early=False, path_sim=sim_path, path_spectra=Pee_path, path_params=params_path, path_ion=ion_path,
                                    LoReLi_format,True, verbose=True)

        if np.isnan(ksz.utils.find_index(sim_check.xe)):
            print(f'Sim {sn} is missing redshifts! Skipping...')
            continue

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

        print('skipping Gorce for the moment (the model, never the real the thing!)')
        #sim = Cat(sn, verbose=True)
        # Gorce = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=False,
        #             Pee_data=None, xe_data=sim.xe, z_data=sim.z, k_data=None, alpha0=alpha0, kappa=kappa,
        #             kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
        
        print()
        print('simulating LoReLi spectrum...')
        print('----------------------------')
        
        # LoReLi = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
        #             Pee_data=sim.Pee, xe_data=sim.xe, z_data=sim.z, k_data=sim.k, alpha0=alpha0, kappa=kappa,
        #             kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
        
        LoReLi_smoothed = ksz.KSZ.get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=True,
                    Pee_data=Pee, xe_data=sim.xe, z_data=sim.z, k_data=k, alpha0=KSZ_params['alpha0'],
                    kappa=KSZ_params['kappa'],
                    kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False)
        
        print()
        
    #  np.savez(Gorce_fn, ells=ells, kSZ=Gorce)
        np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi_smoothed)
        #np.savez(LoReLi_fn, ells=ells, kSZ=LoReLi)

        print(f'saving spectra for simulation {sn}...')
        end_time = time.time()
        print(f"One kSZ run took {(end_time - start_time) / 60.0 :.3f} minutes")
        print(f'{j+1} sims completed, {len(sims)-j} to go!')

if __name__ == "__main__":
    main()
