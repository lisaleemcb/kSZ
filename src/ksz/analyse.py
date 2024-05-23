import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import emcee
import corner

import ksz.analyse
import ksz.utils
import ksz.Pee

from scipy.interpolate import CubicSpline
from ksz.parameters import *


def log_like(params, data, model_func, priors, errors_obs, debug=False):
    model_params = cp.deepcopy(modelparams_Gorce2022)
    keys = list(model_params.keys())[:len(params)]

    if debug:
        print('fitting', keys)

    for i, p in enumerate(params):
        key = keys[i]
        # print(f'rewriting {key} with value {p}')
        if key == 'alpha_0':
             model_params[keys[i]] = 10**p
        else:
            model_params[keys[i]] = p

    log_alpha = np.log10(model_params['alpha_0'])

    prior_alpha, prior_kappa, prior_kf, prior_g = priors
    if (log_alpha < prior_alpha[0]) or (log_alpha > prior_alpha[1]):
        if debug:
            print('outside prior range for log_alpha')
        return -np.inf
    if (model_params['kappa'] < prior_kappa[0]) or (model_params['kappa'] > prior_kappa[1]):
        if debug:
            print('outside prior range for kappa')
        return -np.inf
    if (model_params['k_f'] < prior_kf[0]) or (model_params['k_f'] > prior_kf[1]):
        if debug:
            print('outside prior range for k_f')
        return -np.inf
    if (model_params['g'] < prior_g[0]) or (model_params['g'] > prior_g[1]):
        if debug:
            print('outside prior range for g')
        return -np.inf

    # print(model.calc_spectra(model_params).flatten() / data.flatten())
    _model = model_func(model_params).flatten()
    _data = data.flatten()

    if debug:
        fig, ax = plt.subplots(1,2)
        ax[0].plot((_model - _data)**2)
        ax[1].plot((_model - _data)**2 / errors_obs**2)

    return -.5 * np.sum((_model - _data)**2 / errors_obs**2)

def make_errorbars():
    pass


class Fit:
    def __init__(self,
                zrange,
                krange,
                model_params,
                sim,
                priors,
                model_type=ksz.Pee.Gorce2022,
                fit_early=False,
                fit_late=False,
                frac_err=None,
                nwalkers=10,
                ndim=2,
                burnin=100,
                nsteps=int(1e4),
                initialise=True,
                verbose=True,
                debug=False):

        self.k0 = krange[0]
        self.kf = krange[1]
        self.model_params = model_params
        self.sim = sim
        self.priors = priors
        self.model_type = model_type
        self.fit_early = fit_early
        self.fit_late = fit_late
        self.frac_err = frac_err
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.burnin= burnin
        self.nsteps = nsteps
        self.initialise = initialise
        self.verbose = verbose
        self.debug = debug

        if isinstance(zrange, int):
            if debug:
                print('Only one redshift provided')
            self.single_z = True
        else:
            self.single_z = False

        self.k = self.sim.k[self.k0:self.kf]
        if self.single_z:
            self.zi = zrange
            self.z = self.sim.z[self.zi]
            self.xe = np.array([self.sim.xe[self.zi]])
            self.data = self.sim.Pee[self.zi]['P_k'][self.k0:self.kf]
            self.model = self.model_type(self.k, [self.z], self.xe,
                                    model_params=self.model_params,
                                    verbose=self.verbose)
            self.obs_errs = self.sim.Pee[self.zi]['var'][self.k0:self.kf]

        elif not self.single_z:
            self.z0 = zrange[0]
            self.zf = zrange[1]
            self.z = self.sim.z[self.z0:self.zf]
            self.xe = self.sim.xe[self.z0:self.zf]
            self.data = ksz.utils.unpack_data(sim.Pee, 'P_k', zrange, krange)
            self.model = self.model_type(self.k, self.z, self.xe,
                                    model_params=self.model_params,
                                    verbose=self.verbose)
            self.obs_errs = np.sqrt(ksz.utils.unpack_data(sim.Pee, 'var', zrange, krange))


        if self.fit_early == self.fit_late:
            if debug:
                print('Fitting the full electron power spectrum')
            self.model_func = self.model.calc_spectra

        if self.fit_early:
            if self.verbose:
                print('Fitting the early time Pee')
            self.model_func = self.model.earlytime
            self.Pbb = ksz.utils.unpack_data(sim.Pbb, 'P_k', zrange, krange)
            self.data = self.data - self.Pbb
           #  self.data = self.model.earlytime(model_params)

        if self.fit_late:
            if self.verbose:
                print('Fitting the late time Pee')
            self.data = self.sim.Pee[self.zi]['P_k'][self.k0:self.kf]
            self.model_func = self.model.latetime

        if self.frac_err is not None:
            self.obs_errs = cp.copy(self.data) * self.frac_err
        else:
            if self.verbose:
                print('Using sample variance for errors')

        self.truths = cp.copy(list(self.model_params.values()))
        self.truths[0] = np.log10(self.truths[0])

        if initialise:
            if verbose:
                print('initialisation requested. running fit...')
            self.samples, self.logp = self.run_fit()
            max_logp = np.argmax(self.logp)
            self.logp_truths = self.samples[max_logp]
            self.std = self.samples.std(axis=0)
            self.fit_params = ksz.utils.pack_params(self.logp_truths)
            self.fit_spectra = self.model.calc_spectra(model_params=self.fit_params)

    def run_fit(self):
        if self.verbose or self.debug:
            print('initialising fit...')
        if self.debug:
            print(f'fit params are: {self.truths[:self.ndim]}')

        p0 = np.random.normal(scale=.001, size=(self.nwalkers, self.ndim)) * np.asarray(self.truths[:self.ndim])
        p0 = p0 + np.ones_like(p0) * np.asarray(self.truths[:self.ndim])

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, ksz.analyse.log_like,
                                        args=[self.data, self.model_func, self.priors, self.obs_errs.flatten()])

        state = sampler.run_mcmc(p0, self.burnin)
        sampler.reset()
        if self.debug:
            print('finished burn in phase, getting started on run...')

        sampler.run_mcmc(state, self.nsteps, progress=True)

        if self.debug:
            print('fetching samples and posterior values...')
        samples = sampler.get_chain(flat=True)
        logp = sampler.get_log_prob(flat=True)



        return samples, logp
