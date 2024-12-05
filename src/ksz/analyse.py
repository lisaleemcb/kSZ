import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import emcee

import ksz.analyse
import ksz.utils
import ksz.Pee

from scipy.interpolate import CubicSpline
from ksz.parameters import *


def lklhd(pvals, data, model_func, priors, obs_errs, pfit, debug=False):
    # technically the ln likelihood
    params = cp.deepcopy(modelparams_Gorce2022)

    if debug:
        print('fitting', pfit)
        print('data:', data)

    for i, p in enumerate(pfit):
        params[p] = pvals[i]

    if debug:
        print('params:', params)

    for i, p in enumerate(pvals):
        prior_range = priors[i]

        if (p <= prior_range.min()) | (p >= prior_range.max()):
            if debug:
                print(f'outside prior range with {model.keys()[i]}={p}')  
            return -np.inf


    # print(model.calc_spectra(model_params).flatten() / data.flatten())
    _model = model_func(params).flatten()
    _data = data.flatten()
    _errs = obs_errs.flatten()

    if debug:
        fig, ax = plt.subplots(1,2)
        ax[0].plot((_model - _data)**2)
        ax[1].plot((_model - _data)**2 / _errs**2)

    return -.5 * np.sum((_model - _data)**2 / _errs**2)

def chi2_contributions(pvals, data, model_func, priors, obs_errs, pfit, debug=False):
    # technically the ln likelihood
    params = cp.deepcopy(modelparams_Gorce2022)
    for i, p in enumerate(pfit):
        params[p] = pvals[i]

    model = model_func(params)
    
    # hardcoded since data is shape (zsize, ksize)
    chi2_z = np.zeros_like(data.shape[0])
    chi2_k = np.zeros_like(data.shape[1])

    for i in range(chi2_z.size):
        chi2_z[i] = (model[i] - data[i])**2 / obs_errs[i]**2
    for i in range(chi2_k.size):
        chi2_k[i] = (model[:,i] - data[:,i])**2 / obs_errs[:,i]**2

    return chi2_z, chi2_k

def make_errorbars():
    pass

def make_priors(params, width=.75):
    pvals = cp.deepcopy(np.asarray(list(params.values())))
    prior_bounds = width * pvals[:,None] * np.ones((pvals.size, 2))
    prior_bounds[:,0] = -1 * prior_bounds[:,0]

    prior_bounds = pvals[:,None] + prior_bounds

    return prior_bounds

def lklhd_EMMA(params,
               data,
               model_func,
               priors,
               obs_errs):

    n_copies = data.shape[0]
    lklhds = np.zeros(n_copies)

    for i in range(n_copies):
        lklhds[i] = lklhd(params,
                data[i],
                model_func,
                priors,
                obs_errs[i].flatten(),
                debug=False)

    return lklhds.sum()

class Fit:
    def __init__(self,
                zrange,
                krange,
                params,
                sim,
                priors=None,
                data=None,
                obs_errs=None,
                frac_err=None,
                load_errs=True,
                model_type=ksz.Pee.Gorce,
                Pdd=None,
                pfit=['alpha0', 'kappa'],
                fit_early=False,
                fit_late=False,
                fit_EMMA=False,
                nwalkers=10,
                burnin=100,
                nsteps=int(1e4),
                initialise=True,
                verbose=False,
                debug=False):

        self.krange = krange
        self.zrange = zrange
        self.params = params
        self.sim = sim
        self.frac_err = frac_err
        self.load_errs = load_errs
        self.model_type = model_type
        self.Pdd = Pdd
        self.pfit = pfit
        self.fit_early = fit_early
        self.fit_late = fit_late
        self.fit_EMMA = fit_EMMA
        self.nwalkers = nwalkers
        self.ndim = len(pfit)
        self.burnin= burnin
        self.nsteps = nsteps
        self.initialise = initialise
        self.verbose = verbose
        self.debug = debug

        self.k = self.sim.k[krange]
        self.z = self.sim.z[self.zrange]
        self.xe = self.sim.xe[self.zrange]

        if priors is None:
            self.priors = make_priors(self.params)
        else:
            self.priors = priors

        if self.debug:
            print('priors:', self.priors)

        if self.Pdd is None:
            Pdd = np.load('/Users/emcbride/kSZ/data/Pdd.npy')
            z_inter = np.linspace(5, 25, 100)
            Pdd_spline = CubicSpline(z_inter, Pdd[:,self.krange])
            Pdd_inter = Pdd_spline(self.z)

            self.Pdd = Pdd_inter

        if self.load_errs:
            EMMA_Pee = np.load('/Users/emcbride/kSZ/data/EMMA/Pee_values.npy')
            EMMA_err = np.load('/Users/emcbride/kSZ/data/EMMA/Pee_err.npy')
            EMMA_z = np.loadtxt('/Users/emcbride/kSZ/data/EMMA/zbins.txt')
            EMMA_xe = np.loadtxt('/Users/emcbride/kSZ/data/EMMA/xbins.txt')
            EMMA_k = np.loadtxt('/Users/emcbride/kSZ/data/EMMA/kbins.txt')

            errs_fn = '/Users/emcbride/kSZ/data/EMMA/EMMA_frac_errs.npz'
            errs = np.load(errs_fn)
            EMMA_k = errs['k']
            frac_err_EMMA = errs['err']
            self.err_spline  = CubicSpline(EMMA_k, frac_err_EMMA)

        #==============================
        # initialise data
        #==============================
        if data is None:
            self.data = sim.Pee[np.ix_(zrange, krange)] # ix_ just makes a correct mesh of zrange, krange
        elif data is not None:
            self.data = data

        #==============================
        # initialise model
        #==============================
        self.model = self.model_type(self.k, self.z, self.xe, Pdd=self.Pdd,
                                model_params=self.params,
                                verbose=self.verbose)

        if obs_errs is None:
            self.obs_errs = self.err_spline(sim.k[krange]) * self.model.spectra
        elif obs_errs is not None:
            self.obs_errs = obs_errs

        if self.fit_early == self.fit_late:
            if self.debug:
                print('Fitting the full electron power spectrum')
            self.model_func = self.model.calc_spectra

        if self.fit_early:
            if self.verbose:
                print('Fitting the early time Pee')
            self.model_func = self.model.earlytime
            self.Pbb = sim.Pbb
            self.data = self.data - self.Pbb
           #  self.data = self.model.earlytime(params)

        if self.fit_late:
            if self.verbose:
                print('Fitting the late time Pee')
            self.data = self.sim.Pee[self.zi]['P_k'][self.k0:self.kf]
            self.model_func = self.model.latetime

        if self.frac_err is not None:
            self.obs_errs = cp.deepcopy(self.data) * self.frac_err
        elif not self.load_errs:
            if self.verbose:
                print('Using sample variance for errors')

        if self.initialise:
            if self.verbose:
                print('initialisation requested...')
           # self.samples, self.logp = self.mcmc_fit()
            self.lklhd = self.direct_lklhd_eval()

    def mcmc_fit(self):
        if self.verbose or self.debug:
            print('running mcmc...')

        pvals = np.asarray([self.params[key] for key in self.pfit])
        if self.debug:
            print(f'fit params are: {pvals}')
        p0 = np.random.normal(scale=.01, size=(self.nwalkers, self.ndim))
        p0 = p0 + np.ones_like(p0) * pvals

        if self.fit_EMMA:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, ksz.analyse.lklhd_EMMA,
                                            args=[self.data, self.model_func, self.priors, self.obs_errs, self.pfit])


        elif not self.fit_EMMA:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, ksz.analyse.lklhd,
                                            args=[self.data, self.model_func, self.priors, self.obs_errs.flatten(), self.pfit])

        state = sampler.run_mcmc(p0, self.burnin)
        sampler.reset()
        if self.debug:
            print('finished burn in phase, getting started on run...')

        sampler.run_mcmc(state, self.nsteps, progress=True)

        if self.debug:
            print('fetching samples and posterior values...')

        samples = sampler.get_chain(flat=True)
        logp = sampler.get_log_prob(flat=True)

        self.samples = samples
        self.logp = logp
        self.max_logp = self.logp[np.argmax(self.logp)]
        self.logp_best = self.samples[np.argmax(self.logp)]
        self.std = self.samples.std(axis=0)
        self.mcmc_params = ksz.utils.pack_params(self.logp_best, self.pfit)
        self.mcmc_spectra = self.model.calc_spectra(model_params=self.mcmc_params)

        return samples, logp
    
    def direct_lklhd_eval(self):
        if self.ndim > 2:
            return -np.inf
        a0 = np.linspace(*self.priors[0], 500)
        kappa = np.linspace(*self.priors[1], 600)
        lklhd = np.zeros((a0.size, kappa.size))

        for i, ai in enumerate(a0):
            for j, ki in enumerate(kappa):
                lklhd[i,j] = ksz.analyse.lklhd((ai, ki), self.data, self.model_func,
                                                            self.priors, self.obs_errs, self.pfit)
                
        flat_index = np.argmax(lklhd)
        indices = np.unravel_index(flat_index, lklhd.shape)
       
        self.reduced_chi2 =  np.abs(lklhd[indices]) / (self.data.size - self.ndim)
        self.lklhd_params = cp.deepcopy(self.params)
        self.lklhd_params['alpha0'] = a0[indices[0]]
        self.lklhd_params['kappa'] = kappa[indices[1]]

        return lklhd
    


