import numpy as np
import camb

from ksz.utils import *
from ksz.parameters import *

class Pee:
    def __init__(self,
                k,
                z,
                xe,
                verbose=False):

        self.z = z
        self.k = k
        self.xe = xe

        self.verbose = verbose

class Gorce2022(Pee):
    def __init__(self,
                k,
                z,
                xe,
                Pdd=None,
                model_params=None,
                cosmo_params=None,
                astro_params=None,
                verbose=False):

        super().__init__(k, z, xe, verbose=verbose)

        if self.verbose:
            print('Initialising Gorce2022 parameterisation...')
            print('')
            print(f'z: {z}')
            print(f'k: {k}')
            print(f'xe: {xe}')
            print(f'model params: {model_params}')
            print('')

        if Pdd is not None:
            self.Pdd = Pdd
        elif Pdd is None:
            self.Pdd = self.calc_Pdd()

        if model_params is not None:
            self.model_params = model_params
        elif model_params is None:
            self.model_params = modelparams_Gorce2022

        if cosmo_params is not None:
            self.cosmo_params = cosmo_params
        elif cosmo_params is None:
            self.cosmo_params = cosmoparams_LoReLi

        if astro_params is not None:
            self.astro_params = astro_params
        elif astro_params is None:
            self.astro_params = astro_fiducial

        self.spectra = self.calc_spectra()

    def Pee(self, z, model_params=None):
        if model_params is not None:
            model_params = model_params
        else:
            model_params = self.model_params

        spectra = self.calc_spectra(model_params=model_params)
        return spectra[np.where(self.z == z)[0]].flatten()

    def calc_spectra(self, z=None, model_params=None):
        if z is not None:
            z = z
        else:
            z = self.z

        if model_params is not None:
            model_params = model_params
        else:
            model_params = self.model_params

        # if self.verbose:
        #     print('Calculating spectra with model parameters:', model_params)
        #     print('')

        fH = self.astro_params['fH']
        xe = self.xe[:, np.newaxis]

        return (fH - xe) * self.earlytime(model_params) + xe * self.latetime(model_params)

    def earlytime(self, model_params, z=None, power=(1.0 / 5.0)):
        if z is not None:
            z = z
        else:
            z = self.z

        k = self.k
        k = k[:, np.newaxis]
        xe = self.xe

        alpha_0 = model_params['alpha_0']
        kappa = model_params['kappa']

        # transposing so that the output is of the shape (z.size, k.size)
        return (alpha_0 * xe**(-power))[:, np.newaxis] / (1 + (k/kappa)**3 * xe).T

    def latetime(self, model_params, z=None, k=None):
        if z is not None:
            z = z
        else:
            z = self.z

        if k is not None:
            k = k
        else:
            k = self.k

        return self.b_deltae(model_params)**2 * self.Pdd

    def b_deltae(self, model_params, z=None):
        if z is not None:
            z = z
        else:
            z = self.z

        k = self.k
        k_f = model_params['k_f']
        g = model_params['g']

        return .5 * (np.exp(-k/k_f) + 1 / (1 + (g * k / k_f)**(7.0/2.0)))

    def calc_Pdd(self, z=None, k=None):
        if self.verbose:
            print('Now calculating the dark matter overdensity power spectrum, Pdd...')
            print('')

        if z is not None:
            z = z
        else:
            z = self.z

        if k is not None:
            k = k
        else:
            k = self.k

        H0 = cosmoparams_LoReLi['H0'].value
        h = H0 / 100
        ombh2 = cosmoparams_LoReLi['Ob_0'] * h**2
        omch2 = cosmoparams_LoReLi['Om_0'] * h**2

        # CAMB initialisation
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.set_for_lmax(2500, lens_potential_accuracy=0);

        pars.set_matter_power(redshifts=np.asarray(z), kmax=10.0)
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=k[0] * results.Params.h,
                                                        maxkh=k[-1] * results.Params.h,
                                                        npoints= k.size)
        spl = CubicSpline(kh, pk.T)
        Pdd = spl(self.k)

        return Pdd.T

class LoReLi(Pee):
    def __init__(self,
                k,
                z,
                xe,
                model_params,
                cosmo_params,
                astro_params,
                special_feature):
        super().__init__(k, z, xe, model_params, cosmo_params, astro_params)

        self.special_feature = special_feature

    def specific_method(self):
        print("Specific method for Model2")
