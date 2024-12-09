import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner

import ksz
from ksz.parameters import *


def plot_mcmc(fit_instance, truths=None):
    if truths is None:
        truths = fit_instance.truths

    fig = corner.corner(fit_instance.samples, truths=truths[:fit_instance.ndim])

    #return fig

def plot_spectra(z, k, spectra, xe=None, color_scale='z', ax=None):
    if color_scale == 'z':
        norm = matplotlib.colors.Normalize(vmin=3.0, vmax=23)
        cmap = plt.get_cmap('viridis_r')

    if color_scale == 'xe':
        if not xe:
            raise ValueError("I need an ionisation history if you want colors scaled by xe!")
      
        norm = matplotlib.colors.LogNorm(vmin=xe.min(), vmax=1.08)
        cmap = plt.get_cmap('plasma')

    if ax:
        for i in range(z.size):
            if color_scale == 'z':
                cn = z[i]
            if color_scale == 'xe':
                cn = xe[i]

            ax.loglog(k, spectra[i], color=cmap(norm(cn)), label=f'z={z}')

        return ax

    else:
        fig, ax = plt.subplots()

        for i in range(z.size):
            if color_scale == 'z':
                cn = z[i]
            if color_scale == 'xe':
                cn = xe[i]

            plt.loglog(k, spectra[i], color=cmap(norm(cn)))

        return fig

def plot_timeline(sim, fit, fit_vals, nrows, ncols, figsize=(15,12)): 

    import pickle

    # Load the dictionary from a pickle file
    with open('/Users/emcbride/kSZ/notebooks/a0.pkl', 'rb') as f:
        a0s = pickle.load(f)

    with open('/Users/emcbride/kSZ/notebooks/kappa.pkl', 'rb') as f:
        kappas = pickle.load(f)

    fit_params_new = {'alpha0': fit_vals[0],
                    'kappa': fit_vals[1],
                    'a_xe': -0.2,
                    'k_xe': 1.0,
                    'k_f': 9.4,
                    'g': 0.5}
    
    fit_params_old = {'alpha0': a0s[sim.sim_n],
                    'kappa': kappas[sim.sim_n],
                    'a_xe': -0.2,
                    'k_xe': 1.0,
                    'k_f': 9.4,
                    'g': 0.5}
    
    data = sim.Pee
    model_old = ksz.Pee.Gorce2022(sim.k, sim.z, sim.xe,
                                    model_params=fit_params_old, verbose=False).spectra
    model_new = ksz.Pee.Gorce2022(sim.k, sim.z, sim.xe,
                                    model_params=fit_params_new, verbose=False).spectra
    

    LoReLi_kwargs = {'color':'green', 
                     'ls': ':',
                     'lw': 1.0}
    
    new_kwargs = {'color':'deeppink', 
                     'ls': ':',
                     'lw': 1.0}
    
    old_kwargs = {'color':'blue', 
                     'ls': ':',
                     'lw': 1.0}
    
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)

    axes = axes.flatten()


    # data first
    for i in range(sim.z.size):
        ax = axes[i]
        ax.loglog(sim.k, data[i], **LoReLi_kwargs) #, label='LoReLi' if i==0 else "_nolegend_")
        ax.loglog(sim.k, model_old[i], **old_kwargs) #, label='check' if i==0 else "_nolegend_")
        ax.loglog(sim.k, model_new[i], **new_kwargs) #, label='new' if i==0 else "_nolegend_")


        anno = f' z={sim.z[i]:.2f}\nxe={sim.xe[i]:.5f}'
        ax.annotate(anno, (0.05,.05), xycoords='axes fraction', fontsize=9)

    LoReLi_kwargs['lw'] = 1.5
    new_kwargs['lw'] = 1.5
    old_kwargs['lw'] = 1.5

    old_kwargs['ls'] = '-'
    new_kwargs['ls'] = '-'

    LoReLi_kwargs['alpha'] = 1.0
    new_kwargs['alpha'] = 1.0
    old_kwargs['alpha'] = 1.0

    LoReLi_kwargs['ls'] = None
    LoReLi_kwargs['marker'] = '.'

    # fit on top
    check = 1
    for i in fit.zrange:
        ax = axes[i]

        ax.errorbar(sim.k[fit.krange], data[i][fit.krange], **LoReLi_kwargs, 
                    yerr=fit.obs_errs[i-fit.zrange[0]][fit.krange], label='LoReLi' if check==1 else "_nolegend_")
        ax.loglog(sim.k[fit.krange], model_old[i][fit.krange], **old_kwargs, label='old fit' if check==1 else "_nolegend_")
        ax.loglog(sim.k[fit.krange], model_new[i][fit.krange], **new_kwargs, label='new fit' if check==1 else "_nolegend_")

        check = 0
    # for j, spec in enumerate(spectra[3:]):
    #     spec_kw = spectra_kwargs[j]
    #     for i in range(len(labels['z'])):
    #         ax = axes[i]
    #         ax.loglog(k, spec[i], **spec_kw) #, yerr=np.abs(obs_errors)[i],)
    #         label = spectra_names[j] if i == 0 else "_nolegend_"
    #         spec_kw['label'] = label

    fig.supxlabel('k', y=0.05, fontsize=16)
    # fig.supylabel('Power')
    fig.legend(fontsize=16)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'Simulation {sim.sim_n}', fontsize=20)
    fig.subplots_adjust(top=0.95)

def plot_timeline_reduced(sim, fit, fit_vals, nrows, ncols, figsize=(15,12)): 

    import pickle

    # Load the dictionary from a pickle file
    with open('/Users/emcbride/kSZ/notebooks/a0.pkl', 'rb') as f:
        a0s = pickle.load(f)

    with open('/Users/emcbride/kSZ/notebooks/kappa.pkl', 'rb') as f:
        kappas = pickle.load(f)

    fit_params_new = {'alpha0': fit_vals[0],
                    'kappa': fit_vals[1],
                    'a_xe': -0.2,
                    'k_xe': 1.0,
                    'k_f': 9.4,
                    'g': 0.5}
    
    fit_params_old = {'alpha0': a0s[sim.sim_n],
                    'kappa': kappas[sim.sim_n],
                    'a_xe': -0.2,
                    'k_xe': 1.0,
                    'k_f': 9.4,
                    'g': 0.5}
    
    data = sim.Pee
    model_old = ksz.Pee.Gorce2022(sim.k, sim.z, sim.xe,
                                    model_params=fit_params_old, verbose=False).spectra
    model_new = ksz.Pee.Gorce2022(sim.k, sim.z, sim.xe,
                                    model_params=fit_params_new, verbose=False).spectra
    

    LoReLi_kwargs = {'color':'green', 
                    'ls': ':',
                    'lw': 1.0}
    
    new_kwargs = {'color':'deeppink', 
                    'ls': ':',
                    'lw': 1.0}
    
    old_kwargs = {'color':'blue', 
                    'ls': ':',
                    'lw': 1.0}
    
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)

    axes = axes.flatten()

    LoReLi_kwargs['lw'] = 1.5
    new_kwargs['lw'] = 1.5
    old_kwargs['lw'] = 1.5

    old_kwargs['ls'] = '-'
    new_kwargs['ls'] = '-'

    LoReLi_kwargs['alpha'] = 1.0
    new_kwargs['alpha'] = 1.0
    old_kwargs['alpha'] = 1.0

    LoReLi_kwargs['ls'] = None
    LoReLi_kwargs['marker'] = '.'

    # fit on top
    check = 1
    for i, zi in enumerate(fit.zrange):
        ax = axes[i]
        ax.errorbar(sim.k[fit.krange], data[zi][fit.krange], **LoReLi_kwargs, 
                    yerr=fit.obs_errs[i][fit.krange], label='LoReLi' if check==1 else "_nolegend_")
        ax.loglog(sim.k[fit.krange], model_old[zi][fit.krange], **old_kwargs, label='old fit' if check==1 else "_nolegend_")
        ax.loglog(sim.k[fit.krange], model_new[zi][fit.krange], **new_kwargs, label='new fit' if check==1 else "_nolegend_")

        anno = f' z={sim.z[zi]:.2f}\nxe={sim.xe[zi]:.5f}'
        ax.annotate(anno, (0.05,.05), xycoords='axes fraction', fontsize=9)

        check = 0
    # for j, spec in enumerate(spectra[3:]):
    #     spec_kw = spectra_kwargs[j]
    #     for i in range(len(labels['z'])):
    #         ax = axes[i]
    #         ax.loglog(k, spec[i], **spec_kw) #, yerr=np.abs(obs_errors)[i],)
    #         label = spectra_names[j] if i == 0 else "_nolegend_"
    #         spec_kw['label'] = label

    fig.supxlabel('k', y=0.05, fontsize=16)
    # fig.supylabel('Power')
    fig.legend(fontsize=16)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'Simulation {sim.sim_n}', fontsize=20)
    fig.subplots_adjust(top=0.95)

def plot_model(model, cscale='z'):
    if cscale == 'z':
        cmap = cmap_z
        norm = norm_z

    if cscale == 'xe':
        cmap = cmap_xe
        norm = norm_xe

    fig, ax = plt.subplots(1,2, figsize=(10,3))

    ax[0].plot(model.z, model.xe)
    
    for i in range(model.z.size):
        ax[1].loglog(model.k, model.spectra[i], color=cmap(norm(model.z[i])))

def plot_KSZ_class(z,k,KSZ, cscale='z'):

    if KSZ.interpolate_xe:
        xe = KSZ.xe_interpolated(z)

    else:
        xe = KSZ.xe(z)

    if cscale == 'z':
        cmap = cmap_z
        norm = norm_z
        scale = z

    if cscale == 'xe':
        cmap = cmap_xe
        norm = norm_xe
        scale = xe

    if KSZ.interpolate_Pee:
        Pee = KSZ.Pee_interpolated(z[:,None], k)
    else:
        Pee = KSZ.Pee(k, z[:,None])

    fig, ax = plt.subplots(1,2, figsize=(10,3))

    ax[0].plot(z, xe)

    for i in range(z.size):
        ax[1].loglog(k, Pee[i], color=cmap(norm(scale[i])))

    #ax[2].plot(ells, KSZ, color=cmap(norm(scale[i])))

