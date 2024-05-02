import numpy as np
import copy as cp
import emcee
import corner

import ksz.analyse
import ksz.utils
import ksz.Pee

from scipy.interpolate import CubicSpline
from ksz.parameters import *


def log_like(params, data, model, priors, errors_obs):
    model_params = {'alpha_0': 10**(3.93),
                      'kappa': 0.084,
                        'k_f': 9.4,
                          'g': .5}

    keys = list(model_params.keys())[:len(params)]

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
       return -np.inf
    if (model_params['kappa'] < prior_kappa[0]) or (model_params['kappa'] > prior_kappa[1]):
        return -np.inf
    if (model_params['k_f'] < prior_kf[0]) or (model_params['k_f'] > prior_kf[1]):
        return -np.inf
    if (model_params['g'] < prior_g[0]) or (model_params['g'] > prior_g[1]):
        return -np.inf

    # print(model.calc_spectra(model_params).flatten() / data.flatten())
    _model = model.calc_spectra(model_params=model_params).flatten()
    _data = data.flatten()

    return -.5 * np.sum((_model - _data)**2 / errors_obs**2)

def run_fit(zrange,
            krange,
            model_params,
            sim,
            priors,
            model_type=ksz.Pee.Gorce2022,
            frac_err=.35,
            nwalkers=10,
            ndim=2,
            burnin=100,
            nsteps=int(1e4)):

    k0, kf = krange

    if isinstance(zrange, int):
        data = sim.Pee[zrange]['P_k'][k0:kf]
        model = model_type(sim.k[k0:kf], [sim.z[zrange]],
                    np.array([sim.xe[zrange]]), model_params=model_params,
                    verbose=False)
    else:
        z0, zf = zrange
        data = ksz.utils.unpack_data(sim, zrange, krange)
        model = model_type(sim.k[k0:kf], sim.z[z0:zf], sim.xe[z0:zf],
                        model_params=model_params, verbose=False)

    obs_errs = cp.copy(data) * frac_err

    truths = cp.copy(list(model_params.values()))
    truths[0] = np.log10(truths[0])

    p0 = np.random.normal(scale=.001, size=(nwalkers, ndim)) * np.asarray(truths[:ndim])
    p0 = p0 + np.ones_like(p0) * np.asarray(truths[:ndim])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, ksz.analyse.log_like,
                                    args=[data, model, priors, obs_errs.flatten()])

    state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)

    samples = sampler.get_chain(flat=True)
    logp = sampler.get_log_prob(flat=True)

    max_logp = np.argmax(logp)
    logp_truths = samples[max_logp]
    std = samples.std(axis=0)

    fit_params = ksz.utils.pack_params(logp_truths)
    if isinstance(zrange, int):
        fit_spectra = model.calc_spectra(model_params=fit_params).flatten()
    else:
        fit_spectra = model.calc_spectra(model_params=fit_params)

    fit_dict = {
                'zrange': zrange,
                'krange': krange,
                'data': data,
                'fit_spectra': fit_spectra,
                'obs_errs': obs_errs,
                'fit_params': fit_params,
                'samples': samples
                }

    return fit_dict
