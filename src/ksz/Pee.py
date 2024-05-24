import numpy as np
#import camb

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

        self.spectra = self.calc_spectra(self.model_params)

        print('Gorce 2022 Pee spectrum is now initialised!')
        print('')

    def Pee(self, z, model_params=None):
        if model_params is not None:
            model_params = model_params
        else:
            model_params = self.model_params

        spectra = self.calc_spectra(model_params)
        return spectra[np.where(self.z == z)[0]].flatten()

    def calc_spectra(self, model_params):
        # if self.verbose:
        #     print('Calculating spectra with model parameters:', model_params)
        #     print('')

        if len(self.z) == 1:
            return (self.earlytime(model_params) + self.latetime(model_params)).flatten()
        else:
            return self.earlytime(model_params) + self.latetime(model_params)

    def earlytime(self, model_params, z=None, power=(1.0 / 5.0)):
        if z is not None:
            z = z
        else:
            z = self.z

        k = self.k

        # to ensure correct broadcasting
        k = k[np.newaxis,:]
        xe = self.xe[:, np.newaxis]

        fH = self.astro_params['fH']
        alpha_0 = model_params['alpha_0']
        kappa = model_params['kappa']
        a_xe = model_params['a_xe']
        k_xe = model_params['k_xe']

        # should be shape (z.size, k.size)
        return (fH - xe) * (alpha_0 * xe**(a_xe)) / (1 + (k/kappa)**3 * xe**(k_xe))
        # return (alpha_0 * xe**(-power)) / (1 + (k/kappa)**3 * xe)

    def latetime(self, model_params, z=None, k=None):
        if z is not None:
            z = z
        else:
            z = self.z

        if k is not None:
            k = k
        else:
            k = self.k

        xe = self.xe[:, np.newaxis]

        return xe * self.b_deltae(model_params)**2 * self.Pdd

    def b_deltae(self, model_params, z=None):
        if z is not None:
            z = z
        else:
            z = self.z

        k = self.k
        k_f = model_params['k_f']
        g = model_params['g']

        # should be shape (1, k.size)
        b_deltae = (.5 * (np.exp(-k/k_f) + 1 / (1 + (g * k / k_f)**(7.0/2.0)))).reshape(1, k.size)
        return b_deltae

    def calc_Pdd(self, z=None, k=None):
        import camb
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

        results = camb.get_results(pars)

        PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
            hubble_units=False, k_hunit=False, kmax=k[-1],
            var1='delta_cdm',var2='delta_cdm', zmax=z[0])


        # should be shape(z.size, k.size)
        # also careful of the redshift ordering as outputted from CAMB!!!
        Pdd = PK.P(z[::-1], k)[::-1]

        return Pdd

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
