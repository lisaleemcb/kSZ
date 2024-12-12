import numpy as np
from astropy import cosmology, units, constants
import matplotlib
import matplotlib.pyplot as plt

#######################################
########### System settings ###########
#######################################

##########################
####  Helpful things  ####
##########################
box_size= 296.0 # Mpc
k_res = ((2 * np.pi) / box_size, (2 * np.pi * 256) / box_size / 2)
k_bins = np.geomspace(k_res[0], k_res[1], 26)
k_center = []
for i in range(k_bins.size-1):
    k_center.append((k_bins[i] + k_bins[i+1]) / 2)

k_center = np.asarray(k_center)

norm_z = matplotlib.colors.Normalize(vmin=3.0, vmax=22)
cmap_z = plt.get_cmap('viridis_r')

norm_xe = matplotlib.colors.LogNorm(vmin=1.7e-4, vmax=1.16)
cmap_xe = plt.get_cmap('plasma')

colors = ['green', 'blue', 'plum', 'goldenrod', 'midnightblue', 'deeppink', 'tomato']

astro_labels = ['Xray_Lfunc', 'hard_Xray_fraction',
                'gasconversion_timescale', 'log10_Mmin', 'ion_escapefrac_post']

astro_pnames = ['L_X', 'f_X', 'tau', 'Mmin', 'f_esc']
astro_pnames_formatted = [r'$L_X$', r'$f_X$', r'$\tau$', r'$M_{\text{min}}$', r'$f_{\text{esc}}$']

astro_labels_formatted = [
 'Xray \n Lfunc',
 'hard \n Xray \n fraction',
 'gas \n conversion \n timescale',
 'log10_Mmin',                  
 'ion \n escapefrac \n post']

##########################
####  Fitting params  ####
##########################

kmin = k_res[0]
kmax = 1.5 # To avoid the splitting between the LoReLi spectra and Gorce model at high k

xemin = .01
xemax = .98

##########################
#### Cosmo parameters ####
##########################
h = 0.678 #* units.km / units.s / units.Mpc
Om_0 = 0.308
OL_0 = 0.692
Ob_0 = 0.0484
obh2 = Ob_0 * h**2
och2 = (Om_0 - Ob_0) * h**2
s8 = 0.815
T_cmb = 2.7255

cosmology_LoReLi=cosmology.FlatLambdaCDM(H0=h*100,Tcmb0=T_cmb,Ob0=Ob_0,Om0=Om_0)
cosmoparams_LoReLi = {'H0': h * 100,
                    'Om_0': Om_0,
                    'OL_0': OL_0,
                    'Ob_0': Ob_0,
                    's8': s8}

Yp = 0.2453
Xp = 1-Yp
mh = constants.m_n.value #kg
rhoc = cosmology_LoReLi.critical_density0.si.value #kg m-3
nh = Xp*Ob_0*rhoc/mh  # m-3
xe_recomb = 1.7e-4

# T_CMB=2.7260 #K
# T_CMB_uK=T_CMB*1e6

###################
#### Constants ####
###################
s_T = constants.sigma_T.value    # sigma_thomson in SI units [m^2]
c = constants.c.value   # speed of light in SI units [m.s-1]
Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3

#######################################
###### REIONISATION PARAMETERS ########
#######################################

# parameters for reionisation history
asym = True #asymmetric or tanh model for xe(z)
zend = 5.5
zre = 7.
z_early = 20.

# reionisation of Helium
HeliumI = True
HeliumII = False
fH = 1.
if HeliumI:
	not4 = 3.9715 #eta
	fHe = Yp/(not4*(1-Yp))
	fH=1+2*fHe
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

astro_fiducial = {'fH': fH}

##########################
#### Model parameters ####
##########################
# parameters for Pee
alpha0 = 3.7
kappa = 0.10

modelparams_Gorce2022 = {'alpha0': 3.93,
                        'kappa': 0.084,
                        'a_xe': -1.0 / 5.0,
                        'k_xe': 1.0,
                        'k_f': 9.4,
                        'g': .5,
				        'B': 0.0}

KSZ_params = {'alpha0': 3.7,
                'kappa': 0.1,
                'a_xe': -0.2,
                'k_xe': 1.0,
                'k_f': 9.4,
                'g': 0.5,
				'B': 0.0}


#########################################
#### Settings for C_ells computation ####
#########################################
### linear ell range for kSZ C_ells
ells = np.linspace(1,15000, 100)

########################################
#### Integration/precision settings ####
########################################

######################################
########### FIT PARAMETERS ###########
######################################

blobnames = [
    "pksz",
    "hksz",
    "tt_cmb",
    "ee_cmb",
    "te_cmb"
    "tau",
]
# priors
z_max = 20.0
xe_recomb = 1.7e-4

#####################################
############# COSMOLOGY #############
#####################################

T_cmb = 2.7255

#######################################
########### System settings ###########
#######################################

Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3

#######################################
###### REIONISATION PARAMETERS ########
#######################################

# reionisation of Helium
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

########################################
#### Integration/precision settings ####
########################################

# minimal and maximal values valid for CAMB interpolation of
kmax_pk = 1570.5

### Settings for theta integration
num_th = 50
th_integ = np.linspace(0.0001, np.pi * 0.9999, num_th)
mu = np.cos(th_integ)  # cos(k.k')

### Settings for k' (=kp) integration
# k' array in [Mpc-1] - over which you integrate
min_logkp = -5.0
max_logkp = 1.5
dlogkp = 0.05
kp_integ = np.logspace(min_logkp, max_logkp, int((max_logkp - min_logkp) / dlogkp) + 1)
# minimal and maximal values valid for CAMB interpolation of
kmax_camb = 6.0e3
kmin_camb = 7.2e-6
krange_camb = np.logspace(np.log10(kmin_camb), np.log10(kmax_camb), 500)

### Settings for z integration
z_recomb = 1100.0
z_min = 0.10
z_piv = 1.0
z_max = 20.0
dlogz = 0.1
dz = 0.15
z_integ = np.concatenate(
    (
        np.logspace(
            np.log10(z_min),
            np.log10(z_piv),
            int((np.log10(z_piv) - np.log10(z_min)) / dlogz) + 1,
        ),
        np.arange(z_piv + dz, 10.0, step=dz),
        np.arange(10, z_max + 0.5, step=0.5),
    )
)
z3 = np.linspace(0, z_recomb, 10000)


###################################
###### ANALYSIS PARAMETERS ########
###################################

# statistical parameters
CL = 95  # confidence interval
percentile1 = (100 - CL) / 2
percentile2 = CL + (100 - CL) / 2
smoothing = 1.0

# plotting parameters
ylabels = [ 'TT', 'EE', 'TE', 'pkSZ', 'hkSZ']
props = dict(boxstyle="round", facecolor="white", alpha=0.5)
colorlist = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
cmaps = ["Blues", "Oranges", "Greens", "PuRd"]
alphas = [0.5, 0.5, 0.5, 0.9]
smooth = 1.0
