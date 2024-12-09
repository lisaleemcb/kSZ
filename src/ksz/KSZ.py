##############################################
######## Computes kSZ power spectrum #########
### Copyright Stephane Ilic & Adélie Gorce ###
##############################################

### added from https://github.com/adeliegorce/forecast_reion_CMB

import matplotlib.pyplot
import copy as cp
import camb
import warnings
from camb import model
import numpy as np
import multiprocessing
from scipy.integrate import simpson, cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d
from astropy import cosmology, constants

from scipy.interpolate import CubicSpline, BSpline, splrep, RegularGridInterpolator

from ksz.parameters import *

def get_KSZ(ells, interpolate_xe=True, debug=False, interpolate_Pee=False,
            Pee_data=None, xe_data=None, z_data=None, k_data=None, alpha0=alpha0, kappa=kappa,
            kmin=1e-6, kmax=3000, xemin=0.0, xemax=1.16, verbose=True, helium_interp=False):
        #     kmin=kmin, kmax=kmax, xemin=xemin, xemax=xemax):
    
    KSZ = KSZ_power(verbose=verbose, interpolate_xe=interpolate_xe, interpolate_Pee=interpolate_Pee,
                helium_interp=helium_interp, debug=debug, alpha0=alpha0, kappa=kappa,
                Pee_data=Pee_data, xe_data=xe_data, z_data=z_data, k_data=k_data,
                xemin=xemin, xemax=xemax, kmin=kmin, kmax=kmax)
    KSZ.run_camb(force=True)
    KSZ.init_reionisation_history()

    spectra = KSZ.run_ksz(ells=ells, patchy=True, Dells=True)[:,0]

    return spectra

class KSZ_power:
    def __init__(
        self,
        dz_h=1.5,
        zre_h=7.0,
        z_early=20.0,
        asym_h_reion=True,
        alpha0=3.7,
        kappa=0.10,
        Pee_data=None,
        xe_data=None,
        z_data=None,
        k_data=None,
        interpolate_Pee=False,
        interpolate_xe=False,
        helium_interp=False,
        helium=False,
        helium2=False,
        xe_recomb=1.0e-4,
        h=0.6774,
        thetaMC=None,
        T_cmb=2.7255,
        Ob_0=0.049,
        Om_0=0.309,
        A_s=2.139e-9,
        n_s=0.9677,
        kp_integ=kp_integ,
        zmin=0,
        zmax=1100, # if you want a higher redshift than Recombination...you shouldn't.
        xemin = 1e-10,
        xemax = 1.16,
        kmin=1e-05,
        kmax=3000.0, # because |k-k'| can be large
        verbose=False,
        debug=False,
        run_CMB=False,
        cosmomc=False,
        run_camb=False,
    ):

        """Container for kSZ power spectrum models.

        This class computes a kSZ (and TT, TE, EE) power spectrum given
        a set of cosmological and reionisation parameters following the
        method described in Gorce et al. 2020, A&A 640 A90.

        Parameters
        ----------
            dz_h (float)
                Proxy for H reionisation duration.
                If asym parameterisation, dz = zre - zend.
                Else, dz is duration of instantaneous reion.
                Default is 1.5.
            zre_h (float)
                H reionisation midpoint (when xe = 0.50).
                Default is 7.0
            z_early (float)
                Redshift around which the first sources form.
                Default is 20.
            asym_h_reion (boolean)
                Whether the reionisation history use a symmetric (tanh, set
                to False) or asymmetric (power-law, set to True) model.
            alpha0 (float)
                Large-scale amplitude of the electron power spectrum.
                See Gorce et al. 2020 for model.
                Default is 3.7.
            kappa (float)
                Minimum bubble size, in Mpc-1.
                See Gorce et al. 2020 for model.
                Default is 0.10.
            helium (boolean)
                If helium is True, the first reionisation of He is included by
                multiplying the H ionisation level by fH = 1.08.
                Default is True.
            helium2 (boolean)
                If helium2 is True, the second reionisation is He is included
                as a tanh whose parameters are read from the parameters.py
                module.
                Default is True.
            xe_recomb (float)
                IGM ionisation level at z = z_early.
                Default is 1.0e-4.
            h (float)
                Reduced Hubble constant.
                Default is 0.6774.
            thetaMC (float)
                Ratio of the sound horizon to the angular diameter distance
                at decoupling, scaled by 100.
                Default is None.
            Ob_0 (float)
                Baryon density at z = 0.
                Default is 0.049.
            Om_0 (float)
                Matter density at z = 0.
                Default is 0.309.
            A_s (float)
                Initial super-horizon amplitude of curvature perturbations
                at k = 0.05 Mpc-1.
                Default is 2.139e-9.
            n_s (float)
                Scalar spectral index.
                Default is 0.9677.
            verbose (boolean)
                If True, run in verbose mode.
                Default is False.
            run_CMB (boolean)
                If True, the TT, TE and EE CMB power spectra are
                computed by CAMB.
                Default is False.
            cosmomc (boolean)
                Format of the cosmological parameters given as inputs.
                This is defined to match the CosmoMC parameterisation.
                If True, do not provide h but thetaMC instead, give for Ob_0
                its value multiplied by h**2, for Om_0 the value of Oc_0 * h**2
                and logA = exp(A_s) / 1e10 instead of A_s.
                Default is False.
            run_camb (boolean)
                If True, runs CAMB to obtain the matter power spectra.
                Default is False.

        """

        self.verbose = verbose
        self.debug = debug
        self.run_CMB = run_CMB
        self.cosmomc = cosmomc
        self.asym_h_reion = asym_h_reion

        # INITIALISE COSMOLOGY

        # Ensure cosmomc value is consistent with input
        # cosmological parameters.
        assert (h is not None and thetaMC is None) or (
            h is None and thetaMC is not None
        ), "You must input h or theta but not both"
        if thetaMC is not None:
            assert cosmomc, "If using theta, must also use cosmomc syntax"

        self.thetaMC = thetaMC
        self.T_cmb = T_cmb
        self.n_s = n_s

        if cosmomc:
            self.obh2 = Ob_0
            self.och2 = Om_0
            self.logA = A_s
            self.A_s = np.exp(A_s) / 1e10
            if thetaMC is not None:
                if self.thetaMC > 1.0:
                    thetaMC = self.thetaMC / 100.0
                else:
                    thetaMC = self.thetaMC
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    cosmomc_theta=thetaMC, ombh2=self.obh2, omch2=self.och2
                )
                self.H0 = pars.H0
                self.h = pars.h
                if self.verbose:
                    print("h = %.4f" % self.h)
            else:
                self.h = h
                self.H0 = h * 100.0
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    H0=self.h*100, ombh2=self.obh2, omch2=self.och2,
                )
                pars.Reion.redshift = zre_h
                if self.asym_h_reion:
                    pars.Reion.dz = dz_h
                else:
                    pars.Reion.delta_redshift = dz_h
                pars.InitPower.set_params(ns=self.n_s, r=0, As=self.A_s)
                results = camb.get_results(pars)
                self.thetaMC = results.cosmomc_theta()
            self.Ob_0 = self.obh2 / self.h ** 2
            self.Om_0 = (self.och2 + self.obh2) / self.h ** 2
        else:
            self.h = h
            self.H0 = self.h * 100.0
            self.Ob_0 = Ob_0
            self.Om_0 = Om_0
            self.obh2 = self.Ob_0 * self.h ** 2
            self.och2 = (self.Om_0 - self.Ob_0) * self.h ** 2
            self.A_s = A_s
            self.logA = np.log(self.A_s * 1e10)
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=self.h*100, ombh2=self.obh2, omch2=self.och2,
            )
            pars.Reion.redshift = zre_h
            if self.asym_h_reion:
                pars.Reion.dz = dz_h
            else:
                pars.Reion.delta_redshift = dz_h
            pars.InitPower.set_params(ns=self.n_s, r=0, As=self.A_s)
            results = camb.get_results(pars)
            self.thetaMC = results.cosmomc_theta()

        cos = cosmology.FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        self.Yp = 0.2453
        self.nh = (1.0 - self.Yp) * self.Ob_0 \
            * cos.critical_density0.si.value / constants.m_n.value  # m-3

        # INITIALISE REIONISATION
        # interpolation check
        self.interpolate_Pee = interpolate_Pee
        self.interpolate_xe = interpolate_xe
        if self.interpolate_Pee:
            if Pee_data is None:
                raise ValueError("Pee interpolation from data requested, which requires a Pee, xe, z, k, but missing Pee")
            if xe_data is None:
                raise ValueError("Pee interpolation from data requested, which requires a Pee, xe, z, k, but missing xe")
            if z_data is None:
                raise ValueError("Pee interpolation from data requested, which requires a Pee, xe, z, k, but missing z")
            if k_data is None:
                raise ValueError("Pee interpolation from data requested, which requires a Pee, xe, z, k, but missing k")

        if self.interpolate_xe:
            if xe_data is None:
                raise ValueError("xe interpolation from data requested, which requires a xe and z but missing xe")
            if z_data is None:
                raise ValueError("xe interpolation from data requested, which requires a xe and z but missing z")
            # 
            self.helium_interp = helium_interp
            # perform checks
            if np.all(np.diff(z_data) > 0):
                if xe_data[-1] > xe_data[0]:
                    raise ValueError("Your redshift and xe orders are not consistent. \
                                        One is time-ordered earliest to latest and the other is the opposite!") 
                else:
                    self.data_order = 'low to high redshift'

            if np.all(np.diff(z_data) < 0):
                if xe_data[-1] < xe_data[0]:
                    raise ValueError("Your redshift and xe orders are not consistent. \
                                        One is time-ordered earliest to latest and the other is the opposite!") 
                else:
                    self.data_order = 'high to low redshift'

            if debug:
                print('data was inputted from', self.data_order)

            if self.debug:
                # print('================================')
                # print('Data for Interpolation (before ordering)')
                # print('================================')
                # print(f'Pee: {Pee_data}')
                # print(f'xe: {xe_data}')
                # print(f'z: {z_data}')  
                # print(f'k: {k_data}')   

                import matplotlib

                norm = matplotlib.colors.Normalize(vmin=z_data.min(), vmax=z_data.max())
                cmap = matplotlib.pyplot.get_cmap('viridis_r')

                fig, ax = matplotlib.pyplot.subplots(1,3, figsize=(10,3))

                ax = ax.flatten()

                ax[0].plot(z_data, marker='<')
                ax[0].plot(xe_data, marker='*')
                ax[1].plot(z_data, xe_data)

                for i in range(len(z_data)):
                    ax[2].loglog(k_data, Pee_data[i], color=cmap(norm(z_data[i])))

                fig.subplots_adjust(wspace=0.0, hspace=0.0)
                fig.suptitle('Data before ordering')

            self.xe_data = cp.deepcopy(xe_data)
            self.z_data = cp.deepcopy(z_data)
            self.k_data = cp.deepcopy(k_data)
            if interpolate_Pee:
                self.Pee_data = cp.deepcopy(Pee_data)

            # index = 0
            # if debug:
            #     print(f'The data as inputted goes from {self.data_order}.')
            # # Here we enforce that the data is always ordered from smallest to largest redshift
            # if self.data_order == 'low to high redshift':
            #   #  index = self.find_index(xe_data[::-1])
            #   #  if index > 0:
            #     self.xe_data = cp.deepcopy(xe_data)#[:-index]
            #     self.z_data = cp.deepcopy(z_data)#[:-index]
            #     self.k_data = cp.deepcopy(k_data) # need this for Pk
            #     if interpolate_Pee:
            #         self.Pee_data = cp.deepcopy(Pee_data)#[:-index,:]


            # elif self.data_order == 'high to low redshift':
            #    # index = self.find_index(xe_data)
            #     self.xe_data = cp.deepcopy(xe_data)[::-1]
            #     self.z_data = cp.deepcopy(z_data)[::-1]
            #     self.k_data = cp.deepcopy(k_data)
            #     if interpolate_Pee:
            #         self.Pee_data = cp.deepcopy(Pee_data)[::-1,:]
                 
            #     print('Faites attention! Your inputted data was reversed so that the output will be ordered from SMALLEST TO LARGEST REDSHIFTS.')

            if self.debug:
                # print('================================')
                # print('Data for Interpolation (after reordering)')
                # print('================================')
                # print(f'Pee: {self.Pee_data}')
                # print(f'xe: {self.xe_data}')
                # print(f'z: {self.z_data}')  
                # print(f'k: {self.k_data}')             

                import matplotlib

                norm = matplotlib.colors.Normalize(vmin=self.z_data.min(), vmax=self.z_data.max())
                cmap = matplotlib.pyplot.get_cmap('viridis_r')

                fig, ax = matplotlib.pyplot.subplots(1,3, figsize=(10,3))

                ax = ax.flatten()

                ax[0].plot(self.z_data, marker='<')
                ax[0].plot(self.xe_data, marker='*')
                ax[1].plot(self.z_data, self.xe_data)

                for i in range(len(self.z_data)):
                    ax[2].loglog(self.k_data, self.Pee_data[i], color=cmap(norm(self.z_data[i])))

                fig.subplots_adjust(wspace=0.0, hspace=0.0)
                fig.suptitle('Data after ordering')

        # H reionisation
        self.tau = 0.0
        self.xe_recomb = xe_recomb
        self.zre_h = zre_h
        self.dz_h = dz_h
        self.z_early = z_early

        # He reionisation
        self.helium = helium
        self.helium2 = helium2
        self.f = 1.0
        if self.helium:
            self.fHe = self.Yp / (3.9715 * (1 - self.Yp))
            self.f += self.fHe
            if self.helium2:
                self.f += self.fHe
        else:
            self.fHe = 0.0
        if self.verbose:
            print("Late-time ionisation fraction: %.2f" % self.f)

        # KSZ shape parameters
        self.alpha0 = alpha0
        self.kappa = kappa

        # KSZ signal cutoff values
        self.zmin = zmin
        self.zmax = zmax
        self.xemin = xemin
        self.xemax = xemax
        self.kmin = kmin
        self.kmax = kmax

        dlogkp = 0.05
        self.kp_integ = kp_integ
        if self.verbose:
            print(f'min z: {self.zmin}')
            print(f'max z: {self.zmax}')
            print(f'min xe: {self.xemin}')
            print(f'max xe: {self.xemax}')
            print(f'min k: {self.kmin}')
            print(f'max k: {self.kmax}')

        if self.interpolate_xe:
            xe_max = (1 + self.fHe - self.xe_recomb)
            if self.helium_interp == True:
                self.zend_h = np.sort(z3)[np.where(self.xe_interpolated(z3) >= 1.08)[0][-1]]
            else:
                self.zend_h = np.sort(z3)[np.where(self.xe_interpolated(z3) >= 1.0)[0][-1]] #self.z_data.min()
        elif self.asym_h_reion:
            self.zend_h = self.zre_h - self.dz_h
        else:
            self.zend_h = self.zre_h - self.dz_h/2.

        self.alpha = 0.0
        if self.verbose:
            print("zre_h = %.2f, zend = %.2f" % (self.zre_h, self.zend_h))

        # Initialise arrays for kSZ computation
        self.x_i_z_integ = np.zeros(z_integ.size)  # ionisation level of IGM
        self.tau_z_integ = np.zeros(z_integ.size)  # thomson optical depth
        self.eta_z_integ = np.zeros(z_integ.size)  # comoving dist in [Mpc]
        self.detadz_z_integ = np.zeros(z_integ.size)  # Hubble parameter [m]
        self.f_z_integ = np.zeros(z_integ.size)  # growth rate, unit 1
        self.adot_z_integ = np.zeros(z_integ.size)  # in SI units [s-1]
        self.n_H_z_integ = np.zeros(
            z_integ.size
        )  # number density of baryons in SI units [m-3]
        self.Pk_lin_integ = np.zeros(z_integ.size)  # linear matter ps
        self.b_del_e_integ = np.zeros(z_integ.size)  # electrons bias

        # Fill arrays related to reionisation
        self.init_reionisation_history()
        if run_camb:
            self.run_camb()

    def xe(self, z):
        """
        Computes model's reionisation history.

        The redshift-asymmetric parameterisation of xe(z) in Douspis+2015
        and the class parameters are used.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the ionisation history.
        """

        if self.asym_h_reion:
            # H reionisation
            frac = 0.5 * (np.sign(self.zend_h - z) + 1) + 0.5 * (
                np.sign(z - self.zend_h) + 1
            ) * abs((self.z_early - z) / (self.z_early - self.zend_h)) ** self.alpha
        else:
            deltay = 1.5*np.sqrt(1+self.zre_h) * self.dz_h
            xod = ((1+self.zre_h)**1.5 - (1+z)**1.5)/deltay
            frac = (np.tanh(xod)+1.)/2.

        # add first He reionisation if needed
        xe = (1.0 + self.fHe - self.xe_recomb) * frac

        # add second He reionisation
        if self.helium2:
            assert (self.helium), "Need to set both He reionisation "\
                "to True, cannot have HeII without HeI"
            a = np.divide(1, z + 1.0)
            deltayHe2 = (
                1.5
                * np.sqrt(1 + helium_fullreion_redshift)
                * helium_fullreion_deltaredshift
            )
            VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5
            xod2 = (VarMid2 - 1.0 / a ** 1.5) / deltayHe2
            tgh2 = np.tanh(xod2)  # check if xod<100
            xe += (self.fHe - self.xe_recomb) * (tgh2 + 1.0) / 2.0
        # Ensure continuity of the function
        x = np.where(z < self.z_early, xe + self.xe_recomb, self.xe_recomb)

        return x

    def xe2tau(self, z):
        """
        Computes redshift evolution of the model's optical depth.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the optical depth.
        """
        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        z = np.sort(z)

        if self.interpolate_xe:
            xe = np.sort(self.xe_interpolated(z))[::-1]
           # xe = np.sort(self.interpolate_xe(z))[::-1]
        else:
            xe = np.sort(self.xe(z))[::-1]

        integ = constants.c.value * constants.sigma_T.value * self.nh * xe \
            / cos.H(z).si.value * (1+z)**2
        tofz = cumulative_trapezoid(integ[::-1], z, initial=0)[::-1]

        return tofz

    def W(self, k, x):
        """
        Electron power spectrum at early times.

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            x: float, array of floats
                Ionisation fraction at redshifts one want the
                power spectrum at.
        Outputs
        -------
            W: array of floats
                Power spectrum for k and z.
        """
        return 10 ** self.alpha0 * x ** (-0.2) \
            / (1.0 + x * (k / self.kappa) ** 3.0)

    def bdH(self, k, z, kf=9.4, g=0.5):
        """
        Electrons - matter bias after reionisation.

        Use the parameterisation of Shaw+2012.

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the bias at.
            z: float, array of floats
                Redshift to compute the bias at.
            kf: float
                Mode [Mpc-1] where baryon power spectrum starts
                diverging from matter power spectrum.
                Default is 9.4.
            g: float
                Amplitude of the divergence.
                Default is 0.5 (no unit).
        Outputs
        -------
            b: array of floats
                Bias for k and z.
        """

        return 0.5 * (np.exp(-k / kf)
                      + 1.0 / (1.0 + np.power(g * k / kf, 2.0)))
    
    def find_index(self, arr):
        n = len(arr)
        for i in range(n - 1):
            # Check if all subsequent elements are strictly increasing
            if np.all(np.diff(arr[i:]) > 0):
                return i
        return None  # Return None if no such index is found
            
    def Pee_interpolated(self, z_interp, k_interp, method='cubic'):
        """
        Electron overdensity power spectrum.

        Note: Requires to initialise reionisation history and to
        run camb (will do it if it has not been previously done by
        running self.run_camb() and self.init_reionisation_history().)

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            z: float, array of floats
                Redshift to compute the power spectrum at.
        Outputs
        -------
            Pee: array of floats
                Power spectrum for k and z.
        """

        if self.debug:
            print('Now interpolating Pee...')

        xe_interp = self.xe_interpolated(z_interp)
        Pee_shape = (k_interp * xe_interp).shape
        Pee_interp = np.zeros(Pee_shape)

        if self.debug:
            #pass
            print(f'Pee_interp shape is {Pee_interp.shape}')

        xe = self.xe_data
        k = self.k_data

        if self.debug:
            print(f'xe data is {xe}')
            print(f'k data is {k}')

        fit_points = [xe, k]
        values = np.log10(self.Pee_data)

        if self.debug:
            print(f'Pee data is {values}')

        interp = RegularGridInterpolator(fit_points, values, bounds_error=False, fill_value=np.log10(0.0))
        broadcasted_xe = np.broadcast_to(xe_interp, Pee_shape)
        broadcasted_k = np.broadcast_to(k_interp, Pee_shape)

        interp_xe_k = np.stack([broadcasted_xe.flatten(), broadcasted_k.flatten()], axis=1)
        Pee_interp = interp(interp_xe_k, method=method)

        Pee_interp = 10**Pee_interp.reshape(Pee_shape)

        if self.debug:
            print(f'Pee interpolated is {Pee_interp}')

        mask_k = (k_interp > self.kmin) & (k_interp < self.kmax)
        mask_k = mask_k.astype(int)

        mask_z = (z_interp > self.zmin) & (z_interp < self.zmax)
        mask_z = mask_z.astype(int)
     
        mask_xe = (xe_interp > self.xemin) & (xe_interp < self.xemax)
        mask_xe = mask_xe.astype(int)


      #  return np.where(np.isnan(Pee_interp) | (Pee_interp < 0.0), 0.0, Pee_interp)
        return Pee_interp * mask_k * mask_z * mask_xe

    def xe_interpolated(self, z_interp):
        """
        Electron overdensity power spectrum.

        Note: Requires to initialise reionisation history and to
        run camb (will do it if it has not been previously done by
        running self.run_camb() and self.init_reionisation_history().)

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            z: float, array of floats
                Redshift to compute the power spectrum at.
        Outputs
        -------
            xe_interp: array of floats
                
        """

        if self.debug:
            print('Now interpolating xe...')

        z = np.sort(self.z_data)
        xe = np.sort(self.xe_data)[::-1]

        xe_spline = interp1d(z, xe, axis=0, fill_value="extrapolate") #CubicSpline(z, xe, axis=0)

        frac = 1.0 # - self.xe_recomb)

        xe_He = 0
        if self.helium_interp:
            frac = (1.0 + self.fHe - self.xe_recomb)
            # add second He reionisation
            if self.helium2:
                assert (self.helium), "Need to set both He reionisation "\
                    "to True, cannot have HeII without HeI"
                a = np.divide(1, z_interp + 1.0)
                deltayHe2 = (
                    1.5
                    * np.sqrt(1 + helium_fullreion_redshift)
                    * helium_fullreion_deltaredshift
                )
                VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5
                xod2 = (VarMid2 - 1.0 / a ** 1.5) / deltayHe2
                tgh2 = np.tanh(xod2)  # check if xod<100
                xe_He += (self.fHe - self.xe_recomb) * (tgh2 + 1.0) / 2.0
                xe_He = np.where(z_interp < self.z_early, xe_He, 0.0)

        xe_early = np.where(z_interp > z.max(), self.xe_recomb, 0.0)
        xe_reion = frac * np.where((z_interp <= z.max()) & (z_interp >= z.min()), xe_spline(z_interp), 0.0)
        xe_late = np.where(z_interp < z.min(), frac, 0.0)

        # print('xe_early:', xe_early)
        # print('xe_reion:', xe_reion)
        # print('xe_late:', xe_late)
        
        xe_interp = xe_early + xe_reion + xe_late + xe_He
        # the -1 below is totally ad hoc to make sure it doesn't unnecessarily ruin He reion
        if self.helium_interp:
            xe_interp = np.where((z_interp < helium_fullreion_redshift - 1) & (xe_interp <= (1.0 + 2 * self.fHe - self.xe_recomb)), (1.0 + 2 * self.fHe - self.xe_recomb), xe_interp)

        return  xe_interp

    def Pee(self, k, z):
        """
        Electron overdensity power spectrum.

        Note: Requires to initialise reionisation history and to
        run camb (will do it if it has not been previously done by
        running self.run_camb() and self.init_reionisation_history().)

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            z: float, array of floats
                Redshift to compute the power spectrum at.
        Outputs
        -------
            Pee: array of floats
                Power spectrum for k and z.
        """

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()
        if np.sum(self.f_z_integ) == 0:
            if self.verbose:
                print(
                    "Calling Pee without having initialised the matter "
                    "power spectrum. Running CAMB..."
                )
            self.run_camb()

        mask_k = (k > self.kmin) & (k < self.kmax)
        mask_k = mask_k.astype(int)

        mask_z = (z > self.zmin) & (z < self.zmax)
        mask_z = mask_z.astype(int)

        if self.interpolate_xe:
            xe = self.xe_interpolated(z)
            mask_xe = (self.xe_interpolated(z) > self.xemin) & (self.xe_interpolated(z) < self.xemax)
            mask_xe = mask_xe.astype(int)
        else:
            xe = self.xe(z)
            mask_xe = (self.xe(z) > self.xemin) & (self.xe(z) < self.xemax)
            mask_xe = mask_xe.astype(int)


        Pee = (self.f - xe) / self.f * self.W(k, xe) + xe \
            / self.f * self.bdH(k, z) * self.Pk(k, z)
        
        self.mask_k = mask_k
        self.mask_z = mask_z
        return Pee * mask_k * mask_z * mask_xe
    
    def earlytime(self,z,k):
        if self.interpolate_xe:
            return (self.f - self.xe_interpolated(z)) / self.f * self.W(k, self.xe_interpolated(z)) 
        return (self.f - self.xe(z)) / self.f * self.W(k, self.xe(z)) 
        
    def latetime(self,z,k):
        if self.interpolate_xe:
            return self.xe_interpolated(z) / self.f * self.bdH(k, z) * self.Pk(k, z)
        
        return  self.xe(z) / self.f * self.bdH(k, z) * self.Pk(k, z)
        #return  self.Pk(k, z) # self.bdH(k, z) #* self.Pk(k, z)

    def init_reionisation_history(self,):
        """
        Initialise reionisation history.

        Running this method initialises reionisation
        parameters such as self.tau and self.alpha,
        as well as fills reionisation-related arrays
        for kSZ computation.
        """

        if self.asym_h_reion:
            self.alpha = np.log(1.0 / 2.0 / (1.0 + self.fHe)) / np.log(
                (self.z_early - self.zre_h) / (self.z_early - self.zend_h)
            )

        self.tau = self.xe2tau(z3)[0]
        tauf = interp1d(z3, self.xe2tau(z3))  # interpolation

        if self.interpolate_xe:
            self.x_i_z_integ = self.xe_interpolated(z_integ)  # reionisation history
        else:
            self.x_i_z_integ = self.xe(z_integ)  # reionisation history

        self.tau_z_integ = tauf(z_integ)  # thomson optical depth

        if self.verbose:
            print("tau = %.4f" % self.tau)

    def def_camb(self):
        """
        Define CAMB.parameters object for KSZ_power model.
        """

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0, ombh2=self.obh2, omch2=self.och2, TCMB=self.T_cmb
        )  # ,tau=self.tau)
        pars.InitPower.set_params(ns=self.n_s, r=0, As=self.A_s)
        pars.WantTransfer = True
        # pars.Reion.set_tau(self.tau)
        pars.Reion.use_optical_depth = False
        if self.asym_h_reion:
            pars.Reion.dz = self.dz_h
        else:
            pars.delta_redshift = self.dz_h
        pars.Reion.redshift = self.zre_h
        pars.set_dark_energy()

        return pars

    def get_primary_spectra(self, ells=None, results=None):
        """
        Compute primary power spectra for KSZ_power object.

        Parameters
        ----------
            ells: list of floats
                ells to compute the spectra over.
                If len(ells) == 1, use np.arange(0, ells[0]).
                If None, use np.arange(0, 2000).
            results: CAMB.results object
                Can be fed to avoid computing results twice.

        Outputs
        -------
            CMB_Cells: array of floats of shape ((ells.size, 4).
            First column is ells.
            Second column is TT.
            Third column is EE.
            Fourth column is TE.
        """
        if ells is None:
            lmax = 2000
        elif np.size(ells) == 1:
            lmax = ells[0]
        else:
            lmax = int(np.max(ells))

        pars = self.def_camb()
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        if results is None:
            results = camb.get_results(pars)
        else:
            assert isinstance(results, camb.CAMBdata)
        results.calc_power_spectra(pars)

        if self.verbose:
            print(" Computing CMB power spectra...")
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=lmax)
        CL = powers['total']  # ["unlensed_scalar"]
        ls = np.arange(CL.shape[0])
        CMB_Cells = np.c_[
            ls, CL[:, 0], CL[:, 1], CL[:, 3]
        ]  # tt, ee, te
        if ells is None or np.size(ells) == 1:
            return CMB_Cells
        else:
            return np.c_[ells,
                         interp1d(ls, CL[:, 0])(ells),
                         interp1d(ls, CL[:, 1])(ells),
                         interp1d(ls, CL[:, 3])(ells)
                         ]

    def run_camb(self, force=False, return_Pk=False, kmax_pk=kmax_camb, CMB_ells=[10000]):
        """
        Compute matter power spectra and fill related arrays.

        Parameters
        ----------
            force: boolean
                For function to recompute all parameters and
                arrays even if they are already filled.
                Default is False.
            self.kmax_pk: float
                Maximum k used to compute the matter power
                spectra [Mpc-1].
                Default is read from parameters.py.
            CMB_ells: list of int
                List of multipoles used to compute the primary CMB spectra.


        Note: Requires to initialise reionisation history (will do so
        if it has not been previously done by running
        self.init_reionisation_history().)
        """

        global Pk, Pk_lin

        assert self.tau != 0.0, \
            "Need to initialise reionisation history first"
        if (np.sum(self.f_z_integ) != 0) and (force is False):
            if self.verbose:
                print("CAMB already run. "
                      "Set force to True if want to re-run anyway.")
            return
        if self.verbose:
            print("Running CAMB...")

        pars = self.def_camb()
        data = camb.get_background(pars)
        results = camb.get_results(pars)

        # CMB spectra
        if self.run_CMB:
            self.CMB_Cells = self.get_primary_spectra(ells=CMB_ells, results=results)

        ##############################################
        #### Cosmo functions & derived parameters ####
        ##############################################

        # Hubble function (=adot/a) in SI units [s-1] (CAMB gives km/s/Mpc)
        H = np.vectorize(lambda z: results.hubble_parameter(z) / Mpckm)

        # Growth rate f
        f = np.vectorize(
            lambda z: data.get_redshift_evolution(0.1, z, ["growth"]).flatten()
        )
        # Comoving distance / conformal time in Mpc
        D_C = np.vectorize(lambda z: results.comoving_radial_distance(z))
        # Hydrogen number density function in SI units [m-3]
        n_H = lambda z: self.nh * (1.0 + z) ** 3.0

        self.kmax_pk = kmax_pk
        self.eta_z_integ = D_C(z_integ)  # comoving distance to z in [Mpc]
        self.detadz_z_integ = constants.c.value / H(z_integ)  # Hubble parameter in SI units [m]
        self.f_z_integ = f(z_integ)  # growth rate, no units
        self.adot_z_integ = (1.0 / (1.0 + z_integ)) * H(z_integ)  # in SI units [s-1]
        self.n_H_z_integ = n_H(z_integ)  # number density of baryons in SI units [m-3]

        # Linear matter power spectrum P(z,k) in Mpc^3
        assert (self.kmax_pk <= kmax_camb), \
            'k too large for P(k) extrapolation, modify ell_max or z_min'
        interp_l = camb.get_matter_power_interpolator(
            pars,
            nonlinear=False,
            kmax=self.kmax_pk,
            hubble_units=False,
            k_hunit=False,
            zmax=z_max,
            var1=model.Transfer_nonu,
            var2=model.Transfer_nonu,
        )
        self.Pk_lin = np.vectorize(lambda k, z: interp_l.P(z, k))
        # Non-linear matter power spectrum
        interp_nl = camb.get_matter_power_interpolator(
            pars,
            nonlinear=True,
            kmax=self.kmax_pk,
            hubble_units=False,
            k_hunit=False,
            zmax=z_max,
            var1=model.Transfer_nonu,
            var2=model.Transfer_nonu,
        )
        self.Pk = np.vectorize(lambda k, z: interp_nl.P(z, k))

        self.Pk_lin_integ = self.Pk_lin(
            self.kp_integ[:, None], z_integ[:, None, None]
        )  # linear matter power spectrum
        self.check_ps(self.Pk_lin_integ)

        if self.interpolate_Pee:
            self.Pee_check = self.Pee(self.kp_integ[:, None], z_integ[:, None, None])
            self.Pee_integ = self.Pee_interpolated(z_integ[:,None,None], self.kp_integ[:,None])

        else:
            self.Pee_integ = self.Pee(self.kp_integ[:, None], z_integ[:, None, None])
        self.Pk_integ = self.Pk(self.kp_integ[:, None], z_integ[:, None, None])
        self.check_ps(self.Pee_integ, include_zero=True)
        self.check_ps(self.Pk_integ, include_zero=False)
        self.b_del_e_integ = np.sqrt(self.Pee_integ/self.Pk_integ)  # electrons bias

        if return_Pk:
            return cp.deepcopy(self.Pk)

    def Cl_to_Dl(self, ells, Cells):
        """
        Convert dimensionless C_ell to D_ell in muK2.

        Parameters
        ----------
            ells: array of floats
                Angular multipole(s).
            Cells: array of floats
                Corresponding dimensionless angular power Cell.

        Returns
        -------
            Dell: float
                Corresponding Dell.
        """
        # consistency checks
        assert np.size(ells) == np.shape(Cells)[0], \
            "Inputs have different dimensions."
        # convert ells to array if float is given
        ells = np.array(ells)
        Cells = np.array(Cells)

        if Cells.ndim > 1:
            ells = ells[:, None]

        D_ells = (
            ells
            * (ells + 1)
            * Cells
            / 2.0
            / np.pi
            * (self.T_cmb * 1e6) ** 2
        )

        return D_ells

    def get_contributions(self, ell, patchy=True):
        """
        Compute differential kSZ along redshift and scales for given ell.

        Parameters
        ----------
            ell: float
                Angular multipole to compute the contributions at.
            patchy: boolean
                If True, will compute contributions to patchy kSZ only.
                Default is True.
        Outputs
        ------
            k_integrand: 2D array of floats
                Differential kSZ along k, as a function of self.kp_integ.
                self.kp_integ is first column, the integrand is second column.
            z_integrand: 2D array of floats
                Differential kSZ along z, as a function of z_integ.
                z_integ is first column, the integrand is second column.
            Cell: float, optional
                Total Cell.
        """
        # Preliminaries
        if patchy:
            g = z_integ >= self.zend_h

        else:
            g = np.ones(z_integ.size, dtype=bool)
        

        self.k_z_integ = ell / self.eta_z_integ

        # print('ell:', ell)
        # print('eta_z_integ:', self.eta_z_integ)

        # in [Mpc-1]
        self.k_min_kp = np.sqrt(
            self.k_z_integ[:, None, None] ** 2
            + self.kp_integ[:, None] ** 2.0
            - 2.0 * self.k_z_integ[:, None, None] * self.kp_integ[:, None] * mu
        )

        # if (min(k_z_integ.min(),k_min_kp.min())<kmin_camb) or (max(k_z_integ.max(),k_min_kp.max())>kmax_camb):
        #     raise Warning('Extrapolating the matter PK to too small or too large k')
        self.check_ell = ell
        # Compute I_tot1 and I_tot2, in [Mpc^2]
        if self.interpolate_Pee:
            self.Pee_min_kp_check = self.Pee(self.k_min_kp, z_integ[:, None, None]) 
            self.Pee_min_kp = self.Pee_interpolated(z_integ[:,None,None], self.k_min_kp)

        else:
            self.Pee_min_kp = self.Pee(self.k_min_kp, z_integ[:, None, None]) 

        self.Pk_min_kp = self.Pk(self.k_min_kp, z_integ[:, None, None])
        self.Pk_lin_min_kp = self.Pk_lin(self.k_min_kp, z_integ[:, None, None])

        self.check_ps(self.Pee_min_kp)
        self.I_e1 = (self.Pee_min_kp / self.kp_integ[:, None] ** 2.0)
        self.I_e2 = - (
            np.sqrt(self.Pee_min_kp / self.Pk_min_kp)
            * self.b_del_e_integ
            * self.Pk_lin_min_kp
            / self.k_min_kp ** 2
        )
        self.I_e = self.I_e1 + self.I_e2

        ### Compute Delta_B^2 integrand, in [s-2.Mpc^2]
        self.Delta_B2_integrand = (
            self.k_z_integ[:, None, None] ** 3.0
            / 2.0
            / np.pi ** 2.0
            * (self.f_z_integ[:, None, None] * self.adot_z_integ[:, None, None]) ** 2.0
            * self.kp_integ[:, None] ** 3.0
            * np.log(10.0)
            * np.sin(th_integ)
            / (2.0 * np.pi) ** 2.0
            * self.Pk_lin_integ
            * (1.0 - mu ** 2.0)
            * self.I_e
        )

        ### Compute Delta_B^2, in [s-2.Mpc^2]
        self.Delta_B2 = simpson(simpson(self.Delta_B2_integrand, th_integ), np.log10(self.kp_integ))
        # # Compute C_kSZ(ell) integrand, unit 1
        prefac = 8.0 * np.pi ** 2.0 / (2.0 * ell + 1.0) ** 3.0 \
            * (constants.sigma_T.value / constants.c.value) ** 2.0
        z_integrand = (
            prefac
            * (self.n_H_z_integ[g] * self.x_i_z_integ[g] / (1.0 + z_integ[g])) ** 2.0
            * np.exp(-2.0 * self.tau_z_integ[g]) * self.eta_z_integ[g]
            * self.detadz_z_integ[g] * Mpcm ** 3.0
            * simpson(simpson(self.Delta_B2_integrand, th_integ), np.log10(self.kp_integ))[g]
        )
        k_integrand = simpson(
            prefac
            * (self.n_H_z_integ[g, None] * self.x_i_z_integ[g, None] / (1.0 + z_integ[g, None])) ** 2.0
            * np.exp(-2.0 * self.tau_z_integ[g, None]) * self.eta_z_integ[g, None]
            * self.detadz_z_integ[g, None] * Mpcm ** 3.0
            * simpson(self.Delta_B2_integrand[g], th_integ),
            z_integ[g],
            axis=0
        )

        # Compute C_kSZ(ell), no units
        Cell = trapz(z_integrand, z_integ[g])
        self.check_result(Cell)

        return np.c_[self.kp_integ, k_integrand], np.c_[z_integ[g], z_integrand], Cell


    def C_ell_kSZ(self, ell, patchy=True):
        """
        Compute kSZ angular power spectrum for given model at ell.

        Parameters
        ----------
            ell: float
                Angular multipole to compute the spectrum at.
            patchy: boolean
                If True, will compute both the late-time and
                the patchy kSZ power.
                Default is True.
        Outputs
        ------
            Late time power at ell if patchy is False.
            Tuple of patchy, late-time power if patchy is True.
        """
        ### Preliminaries
        # in [Mpc-1]

        self.k_z_integ = ell / self.eta_z_integ

        # in [Mpc-1]
        self.k_min_kp = np.sqrt(
            self.k_z_integ[:, None, None] ** 2
            + self.kp_integ[:, None] ** 2.0
            - 2.0 * self.k_z_integ[:, None, None] * self.kp_integ[:, None] * mu
        )

        # if (min(k_z_integ.min(),k_min_kp.min())<kmin_camb) or (max(k_z_integ.max(),k_min_kp.max())>kmax_camb):
        #     raise Warning('Extrapolating the matter PK to too small or too large k')
        self.check_ell = ell
        # Compute I_tot1 and I_tot2, in [Mpc^2]
        if self.interpolate_Pee:
            self.Pee_min_kp_check = self.Pee(self.k_min_kp, z_integ[:, None, None]) 
            self.Pee_min_kp = self.Pee_interpolated(z_integ[:,None,None], self.k_min_kp)

        else:
            self.Pee_min_kp = self.Pee(self.k_min_kp, z_integ[:, None, None]) 

        self.Pk_min_kp = self.Pk(self.k_min_kp, z_integ[:, None, None])
        self.Pk_lin_min_kp = self.Pk_lin(self.k_min_kp, z_integ[:, None, None])

        self.check_ps(self.Pee_min_kp)
        self.I_e = (self.Pee_min_kp / self.kp_integ[:, None] ** 2.0) - (
            np.sqrt(self.Pee_min_kp / self.Pk_min_kp)
            * self.b_del_e_integ
            * self.Pk_lin_min_kp
            / self.k_min_kp ** 2
        )

        ### Compute Delta_B^2 integrand, in [s-2.Mpc^2]
        self.Delta_B2_integrand = (
            self.k_z_integ[:, None, None] ** 3.0
            / 2.0
            / np.pi ** 2.0
            * (self.f_z_integ[:, None, None] * self.adot_z_integ[:, None, None]) ** 2.0
            * self.kp_integ[:, None] ** 3.0
            * np.log(10.0)
            * np.sin(th_integ)
            / (2.0 * np.pi) ** 2.0
            * self.Pk_lin_integ
            * (1.0 - mu ** 2.0)
            * self.I_e
        )
        ### Compute Delta_B^2, in [s-2.Mpc^2]
        self.Delta_B2 = simpson(simpson(self.Delta_B2_integrand, th_integ), np.log10(self.kp_integ))

        ### Compute C_kSZ(ell) integrand, unit 1
        self.C_ell_kSZ_integrand = (
            8.0
            * np.pi ** 2.0
            / (2.0 * ell + 1.0) ** 3.0
            * (constants.sigma_T.value / constants.c.value) ** 2.0
            * (self.n_H_z_integ * self.x_i_z_integ / (1.0 + z_integ)) ** 2.0
            * self.Delta_B2
            * np.exp(-2.0 * self.tau_z_integ)
            * self.eta_z_integ
            * self.detadz_z_integ
            * Mpcm ** 3.0
        )

        ### Compute C_kSZ(ell), no units
        result = trapz(self.C_ell_kSZ_integrand, z_integ)
        if patchy:
            g = z_integ >= self.zend_h
            result_p = trapz(self.C_ell_kSZ_integrand[g], z_integ[g])
            return self.check_result(result_p), self.check_result(result)
        else:
            return self.check_result(result)

    def run_ksz(self, ells=[3000], n_threads=1, patchy=True, Dells=False):
        """
        Compute kSZ power spectrum for a range of multipoles.

        Parameters
        ----------
            ells: array of floats
                Angular multipoles to compute the spectrum at.
                Default is l=3000 only.
            n_threads: int
                Number of threads to use for multiprocessing.
                Default is 1.
            patchy: boolean
                If True, will compute both the late-time and
                the patchy kSZ power.
                Default is True.
            Dells: boolean
                If True, give the results in terms of
                D(l) = Tcmb**2 * l(l+1)Cl/2/pi.
                Default is False.
        Outputs
        ------
            Array of floats of shape (len(ells), 1) if patchy is
            False (late time power only) or shape (len(ells), 2)
            if patchy is True (late time and patchy power).
        """

        ells = np.array(ells)

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()

        cos = cosmology.FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        self.kmax_pk = np.max(ells) / cos.comoving_distance(z_min).value
        if self.kmax_pk > 1e3:
            if self.verbose:
                print("Need to re-run CAMB for large k-values.")
            self.run_camb(force=True, kmax_pk=self.kmax_pk)
        if np.sum(self.f_z_integ) == 0:
            self.run_camb(kmax_pk=self.kmax_pk)

        if self.verbose:
            print(
                "Computing for %i l on range [%i,%i] with %i threads"
                % (len(ells), np.min(ells), np.max(ells), n_threads)
            )

        C_ells = np.array([self.C_ell_kSZ(ell, patchy) for ell in ells])
        # C_ells = np.array(multiprocessing.Pool(n_threads).map(self.C_ell_kSZ, ells))

        if not Dells:
            return C_ells
        else:
            return self.Cl_to_Dl(ells, C_ells)

    def check_ps(self, ps, include_zero=True):
        """
        Perform basic consistency checks on power spectrum array.

        Parameters
        ----------
        ps : array of floats
            Power spectrum values.
        include_zero: boolean
            Whether zero values are allowed for the ps.
        """
        if (include_zero and np.any(ps < 0.)) or (np.any(ps <= 0.) and not include_zero):
            print(include_zero)
            print(np.any(ps < 0.))
            print(f'first clause {(include_zero and np.any(ps < 0.))}')

            print(not include_zero)
            print(np.any(ps <= 0.))
            print(f'first clause {(np.any(ps <= 0.) and not include_zero)}')
            raise ValueError('Negative values in the power spectrum.')
        if np.isnan(ps).any():
            raise ValueError('NaN values in the power spectrum.')

    def check_result(self, res):
        """
        Perform basic consistency checks on result array.

        Parameters
        ----------
        res: array of floats
            Array of values to check.
        """
        if np.isnan(res).any():
            raise ValueError(f'NaN values: {res}.')
        return res
