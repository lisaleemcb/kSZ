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


########################################
#### Integration/precision settings ####
########################################
