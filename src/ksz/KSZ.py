import numpy as np

from ksz.utils import *
from ksz.parameters import *

class KSZ:
    def __init__(self,
                k,
                z,
                xe,
                model_params,
                cosmo_params,
                astro_params):

        self.z = z
        self.k = k
        self.xe = xe

        self.model_params = model_params
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params

        def Cell(self):
            z = self.z
            Dimless_Be = dimless(P_Be())


        def P_Be(self):
           pass


class Gorce2022(KSZ):
    def __init__(self,
                k,
                z,
                xe,
                P_m,
                model_params,
                cosmo_params,
                astro_params):

        super().__init__(k, z, xe, model_params, cosmo_params, astro_params)

        self.P_m = P_m

        def Pee(self):
            xe = self.xe

            return (fH - xe) * self.earlytime() + xe * self.latetime()

        def earlytime(self, power=(1.0 / 5.0)):
            xe = self.xe
            k = self.k
            alpha_0 = self.model_params['alpha_0']
            kappa = self.model_params['kappa']

            return (alpha_0 * xe**(-power)) / (1 + (k/kappa)**3 * xe)

        def latetime(self):
            return self.b_deltae(k, z)**2 * self.P_deltadelta(k, z)

        def b_deltae(self, k, z):
            k_f = self.model_params['k_f']
            g = self.model_params['g']

            return .5 * (np.exp(-k/k_f) + 1 / (1 + (g * k / k_f)**(7.0/2.0)))

        def P_deltadelta(self, k, z):
            return self.P_m


class LoReLi(KSZ):
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
