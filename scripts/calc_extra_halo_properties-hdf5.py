# module load StdEnv/2023 python/3.13
# source /scratch/aspadawe/manhattan_suite/observables/pyenvs/mansuite-obs/bin/activate

# import sys
import os
import argparse
import h5py
# import gc
# import copy
# from timeit import default_timer as timer

# import yt
import unyt
# import caesar
# from caesar.hydrogen_mass_calc import get_aperture_masses
import numpy as np

# import pprint

# gc.isenabled()



parser = argparse.ArgumentParser(prog='track_halo_properties.py', description='Track properties of halo and central galaxy across snapshots using supplied progenitors/descendants.')
parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
# parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
#                     help='Whether to clear the output file initially before writing to it')

parser.add_argument('--halo_types', action='store', nargs='*', type=str, default=[],
                    help='Names of halo types for which to calculate properties')
parser.add_argument('--central_types', action='store', nargs='*', type=str, default=[],
                    help='Names of central types for which to calculate properties')

parser.add_argument('--sim_model', action='store', type=str, choices=['Simba', 'Simba-C', 'Obsidian'], required=True,
                    help='Galaxy formation model of the simulation (for black hole feedback criteria)')
parser.add_argument('--mgas', action='store', type=float, required=True,
                    help='Fiducial gas particle mass of the simulation')
parser.add_argument('--mgas_units', action='store', type=str, required=False, default='Msun',
                    help='Units of fiducial gas particle mass of the simulation')
args = parser.parse_args()

# print(args.n_most)


# if not os.path.exists(args.output_file):
#     print('Making output path and file')
#     os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
#     f = h5py.File(args.output_file, 'a')
#     f.close()
#     print()

# if args.clear_output_file:
#     print('Clearing output file')
#     # save_object_with_dill(init_dict, args.output_file, mode='wb')
#     f = h5py.File(args.output_file, 'w')
#     f.close()
#     # f = open(args.output_file, 'w')
#     # f.close()
#     print()





def create_group(file, group):
    if group not in file:
        file.create_group(group)

def create_dataset(group, dataset, shape=(0,), maxshape=(None,), dtype='f8', units=None):
    # print('start')
    print(group, dataset)
    if dataset not in group:
        # print(group, dataset)
        group.create_dataset(dataset, shape=shape, maxshape=maxshape, dtype=dtype)
    group[dataset].attrs['units'] = units
    # print('finish')

def append_to_dataset(group, dataset, value, length):
    if length == len(group[dataset][:]):
        # new_shape = group[dataset].shape
        # # new_shape[0] += 1
        # new_shape = (new_shape[0]+1, *new_shape[1:])
        # new_shape = (*new_shape,)
        new_shape = (group[dataset].shape[0]+1,)
        # print(new_shape)
        group[dataset].resize(new_shape)
    group[dataset][-1] = value




###########################################################################################
## Set up halo/galaxy properties to save ##################################################


# halo_types = ['first_mmp_halo', 'second_mmp_halo', 'first_mmp_central_halo', 'second_mmp_central_halo']
# central_types = ['first_mmp_central', 'second_mmp_central', 'first_mmp_halo_central', 'second_mmp_halo_central']

# halo_types = []
# central_types = []
# if args.track_halos is False and args.track_centrals is False:
#     print('Not tracking halos or centrals - exiting')
#     sys.exit()
# if args.track_halos:
#     halo_types += ['first_mmp_halo']
#     central_types += ['first_mmp_halo_central']
# if args.track_centrals:
#     halo_types += ['first_mmp_central_halo']
#     central_types += ['first_mmp_central']




Lsun = unyt.unyt_quantity(3.8270*(10**33), 'erg/s')




part_types = ['PartType0', 'PartType1', 'PartType2', 'PartType4', 'PartType5']
# apertures = ['30 kpccm', '50 kpccm']
# aperture_names = ['30ckpc', '50ckpc']

mgas = unyt.unyt_quantity(args.mgas, args.mgas_units)
print(f'\nUsing gas particle mass: {mgas}\n')


delta_values = ['2500', '500', '200']
virial_quantities = ['circular_velocity', 'spin_param', 'temperature']
halo_sfr_types = ['', '_100']
# halo_mass_types = ['gas', 'stellar', 'baryon', 'dm', 'dm2', 'dust', 'bh', 'H2', 'H2_ism', 'HI', 'HI_ism']
halo_mass_types = ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dm', 'dm2', 'dust', 'gas', 'stellar', 'total']
# halo_mass_subtypes = ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dm', 'dm2', 'dust', 'gas', 'stellar']
halo_radii_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
halo_radii_XX = ['half_mass', 'r20', 'r80']
halo_metallicity_types = ['mass_weighted', 'sfr_weighted', 'stellar', 'mass_weighted_cgm', 'temp_weighted_cgm']
halo_velocity_dispersion_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# halo_rotation_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# halo_rotation_XX = ['L', 'ALPHA', 'BETA', 'BoverT', 'kappa_rot']
halo_age_types = ['mass_weighted', 'metal_weighted']
halo_temperature_types = ['mass_weighted', 'mass_weighted_cgm']#, 'temp_weighted_cgm']
halo_local_density_types = ['300', '1000', '3000']


# central_mass_types = ['gas', 'stellar', 'bh', 'dust', 'HI', 'H2']#'dm',
# central_mass_apertures = ['', '_30kpc']
central_mass_types = ['H2', 'H2_30kpc', 'H2_ism', 'HI', 'HI_30kpc', 'HI_ism', 'baryon', 'bh', 'bh_30kpc', 'bh_stellar_half_mass_radius', 'dm2_30kpc', 'dm2_stellar_half_mass_radius', 'dm_30kpc', 'dm_stellar_half_mass_radius', 'dust', 'gas', 'gas_30kpc', 'gas_stellar_half_mass_radius', 'star_30kpc', 'star_stellar_half_mass_radius', 'stellar', 'total']
central_mass_total_types = ['total', 'gas', 'dm_30kpc', 'gas_30kpc', 'dm_stellar_half_mass_radius', 'gas_stellar_half_mass_radius']
central_mass_subtypes = [
    ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dust', 'gas', 'stellar', 'total'], ## relative to 'total'
    ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dust', 'gas', 'stellar'], ## relative to 'gas'
    ['H2_30kpc', 'HI_30kpc', 'bh_30kpc', 'dm2_30kpc', 'gas_30kpc', 'star_30kpc', 'dm_30kpc'], ## relative to 'dm_30kpc'
    ['H2_30kpc', 'HI_30kpc', 'bh_30kpc', 'dm2_30kpc', 'gas_30kpc', 'star_30kpc', 'dm_30kpc'], ## relative to 'gas_30kpc'
    ['bh_stellar_half_mass_radius', 'dm2_stellar_half_mass_radius', 'gas_stellar_half_mass_radius', 'star_stellar_half_mass_radius', 'dm_stellar_half_mass_radius'], ## relative to 'dm_stellar_half_mass_radius'
    ['bh_stellar_half_mass_radius', 'dm2_stellar_half_mass_radius', 'gas_stellar_half_mass_radius', 'star_stellar_half_mass_radius', 'dm_stellar_half_mass_radius'], ## relative to 'dgasstellar_half_mass_radius'
]
# central_mass_fof_subtypes = ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dust', 'gas', 'stellar', 'total'] ## relative to 'total'
# central_mass_aperture_subtypes_1 = ['H2_30kpc', 'HI_30kpc', 'bh_30kpc', 'dm2_30kpc', 'gas_30kpc', 'star_30kpc', 'dm_30kpc'] ## relative to 'dm_30kpc'
# central_mass_aperture_subtypes_2 = ['bh_stellar_half_mass_radius', 'dm2_stellar_half_mass_radius', 'gas_stellar_half_mass_radius', 'star_stellar_half_mass_radius', 'dm_stellar_half_mass_radius'] ## relative to 'dm_stellar_half_mass_radius'
central_radii_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_radii_XX = ['half_mass', 'r20', 'r80']
central_sfr_types = ['', '_100']
central_metallicity_types = ['mass_weighted', 'sfr_weighted', 'stellar']
central_velocity_dispersion_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_age_types = ['mass_weighted', 'metal_weighted']
central_temperature_types = ['mass_weighted', 'mass_weighted_cgm']#, 'temp_weighted_cgm']
# central_rotation_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# central_rotation_XX = ['L', 'ALPHA', 'BETA', 'BoverT', 'kappa_rot']


central_star_masses_for_sfr = ['stellar', 'star_30kpc', 'star_stellar_half_mass_radius']
central_star_names_for_sfr = ['', '_30ckpc', '_stellar_half_mass_radius']



f_rad = 0.1
if args.sim_model.lower() == 'simba':
    # Simba jet feedback criteria
    bh_mass_jet_min = unyt.unyt_quantity(4e7, 'Msun')
    bh_fedd_jet_max = unyt.unyt_quantity(0.2, '1')
elif args.sim_model.lower() == 'simba-c':
    # Simba-C jet feedback criteria
    bh_mass_jet_min = unyt.unyt_quantity(7e7, 'Msun')
    bh_fedd_jet_max = unyt.unyt_quantity(0.2, '1')
elif args.sim_model.lower() == 'obsidian':
    # Obsidian jet feedback criteria
    bh_mass_jet_min = unyt.unyt_quantity(5e7, 'Msun')
    bh_fedd_jet_max = unyt.unyt_quantity(0.03, '1')




## Apertures
halo_aperture_names = ['3ckpc', '30ckpc', '50ckpc'] + [f'r{delta_value}c' for delta_value in delta_values] + ['0.5r200c', '0.1r200c']
central_aperture_names = ['3ckpc', '30ckpc', '50ckpc']


## Gas phase definitions
gas_phases = {
    'Sokolowska+2018':[
        'cold',
        'warm',
        'warm_hot',
        'hot',
    ],
    'vandeVoort+2011':[
        'diffuse_IGM',
        'cold_halo_gas',
        'WHIM',
        'ICM',
        'star_forming_ISM',
        'sub_virial',
        'super_virial',
    ],
    'dave+2019':[
        'sub_virial',
        'super_virial',
    ],
    'Aviv':[
        'coupled_gas',
        'decoupled_gas',
        'launched_wind',
        'unlaunched_wind',
        'outflow>0',
        'outflow>0_nowind',
        'outflow>0_onlywind',
        'inflow>0',
        'inflow>0_nowind',
        'inflow>0_onlywind',
        'outflow>1000',
        'outflow>1000_nowind',
        'outflow>1000_onlywind',
        'inflow>1000',
        'inflow>1000_nowind',
        'inflow>1000_onlywind',
        'outflow>5000',
        'outflow>5000_nowind',
        'outflow>5000_onlywind',
        'inflow>5000',
        'inflow>5000_nowind',
        'inflow>5000_onlywind',
        'outflow>10000',
        'outflow>10000_nowind',
        'outflow>10000_onlywind',
        'inflow>10000',
        'inflow>10000_nowind',
        'inflow>10000_onlywind',
        'outflow0-300',
        'outflow0-300_nowind',
        'outflow0-300_onlywind',
        'inflow0-300',
        'inflow0-300_nowind',
        'inflow0-300_onlywind',
        'outflow300-1000',
        'outflow300-1000_nowind',
        'outflow300-1000_onlywind',
        'inflow300-1000',
        'inflow300-1000_nowind',
        'inflow300-1000_onlywind',
        'outflow1000-10000',
        'outflow1000-10000_nowind',
        'outflow1000-10000_onlywind',
        'inflow1000-10000',
        'inflow1000-10000_nowind',
        'inflow1000-10000_onlywind',
        'IGrM',
    ],
}


###########################################################################################




## A few extra properties that can be calculated from those already saved
print('\nCalculating a few extra properties...\n')
with h5py.File(args.output_file, 'r+') as f:
    
    for halo_type in args.halo_types:
        print(f'\n\nCalculating properties for halo type: {halo_type}\n')
        try:
            group = f[f'/{halo_type}']
        except:
            print(f'\n{halo_type} does not exist!\n')
            continue

        ## Scale factor
        try:
            create_dataset(group, f'a', shape=(group[f'z'].shape[0],), dtype='f8', units=group[f'z'].attrs['units'])
            group[f'a'].resize((group[f'z'].shape[0],))
            z = unyt.unyt_array(group['z'][:], group[f'z'].attrs['units'])
            a = 1./(1. + z)
            group['a'][:] = a
        except Exception as error:
            print(f'\nError calculating {halo_type} a: {error}\n')

        ## Time since last snapshot
        try:
            create_dataset(group, f'delta_t', shape=(group[f'age'].shape[0],), dtype='f8', units=group[f'age'].attrs['units'])
            group[f'delta_t'].resize((group[f'age'].shape[0],))
            values_curr = np.nan_to_num(group[f'age'][:], nan=0.0, copy=True)
            values_prev = np.nan_to_num(np.roll(group[f'age'][:], 1), nan=0.0, copy=True)
            values_prev[0] = 0.0
            age_curr = unyt.unyt_array(values_curr, group[f'age'].attrs['units'])
            age_prev = unyt.unyt_array(values_prev, group[f'age'].attrs['units'])
            delta_t = age_curr - age_prev
            group[f'delta_t'][:] = delta_t.in_units(group[f'age'].attrs['units'])
        except Exception as error:
            print(f'\nError calculating {halo_type} delta_t: {error}\n')

        try:
            create_dataset(group, 'ssfr', shape=(group['sfr'].shape[0],), dtype='f8', units='yr**-1')
            group['ssfr'].resize((group['sfr'].shape[0],))
            ssfr = unyt.unyt_array(group['sfr'], group['sfr'].attrs['units'])/unyt.unyt_array(group['stellar_mass'], group['stellar_mass'].attrs['units'])
            group['ssfr'][:] = ssfr.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {halo_type} ssfr: {error}\n')
            # continue

        try:
            create_dataset(group, 'ssfr_100', shape=(group['sfr_100'].shape[0],), dtype='f8', units='yr**-1')
            group['ssfr_100'].resize((group['sfr_100'].shape[0],))
            ssfr_100 = unyt.unyt_array(group['sfr_100'], group['sfr_100'].attrs['units'])/unyt.unyt_array(group['stellar_mass'], group['stellar_mass'].attrs['units'])
            group['ssfr_100'][:] = ssfr_100.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {halo_type} ssfr_100: {error}\n')
            # continue

        


        ## BH accretion rates and bolometric luminosities
        try:
            create_dataset(group, 'bh_mdot_edd-bad', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_edd-bad'].resize((group['bh_mdot'].shape[0],))
            bh_mdot_edd = unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units'])/unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            group['bh_mdot_edd-bad'][:] = bh_mdot_edd.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_mdot_edd-bad: {error}\n')

        try:
            create_dataset(group, 'bh_mdot_edd', shape=(group['bh_mass'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_edd'].resize((group['bh_mass'].shape[0],))
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson))
            group['bh_mdot_edd'][:] = bh_mdot_edd.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_mdot_edd: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol', shape=(group[f'bh_mdot'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol'].resize((group[f'bh_mdot'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units'])
            group[f'bh_Lbol'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_Lbol: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol_edd', shape=(group[f'bh_mdot_edd'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_edd'].resize((group[f'bh_mdot_edd'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot_edd'], group['bh_mdot_edd'].attrs['units'])
            group[f'bh_Lbol_edd'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_Lbol_edd: {error}\n')

        try:
            create_dataset(group, 'bh_mdot_acc', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_acc'].resize((group[f'bh_mdot'].shape[0],))
            bh_mdot_acc = unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units']) / (1 - f_rad)
            group['bh_mdot_acc'][:] = bh_mdot_acc.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_mdot_acc: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol_acc', shape=(group[f'bh_mdot_acc'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_acc'].resize((group[f'bh_mdot_acc'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            group[f'bh_Lbol_acc'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_Lbol_acc: {error}\n')
            # continue

        try:
            create_dataset(group, 'bh_fedd_acc', shape=(group['bh_mdot_acc'].shape[0],), dtype='f8', units='1')
            group['bh_fedd_acc'].resize((group[f'bh_mdot_acc'].shape[0],))
            bh_mdot = unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            bh_mdot_edd = unyt.unyt_array(group['bh_mdot_edd'], group['bh_mdot_edd'].attrs['units'])
            bh_fedd_acc = bh_mdot/bh_mdot_edd
            group['bh_fedd_acc'][:] = bh_fedd_acc.in_units('1')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_fedd_acc: {error}\n')

        
        try:
            ## From Hirschmann+2014, equation 6 & 7, originally from Churazov+2005
            create_dataset(group, f'bh_Lbol_acc_split', shape=(group[f'bh_mdot_acc'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_acc_split'].resize((group[f'bh_mdot_acc'].shape[0],))
            bh_mdot = unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            bh_Lbol_edd = unyt.unyt_array(group['bh_Lbol_edd'], group['bh_Lbol_edd'].attrs['units'])
            bh_fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
            Lbol = unyt.unyt_array(np.zeros(bh_mdot.shape[0]), 'erg/s')
            # Radiative mode
            mask_rad = bh_fedd >= bh_fedd_jet_max
            Lbol[mask_rad] = ((f_rad/(1-f_rad)) * bh_mdot[mask_rad] * unyt.c**2).in_units('erg/s')
            # Mechanical mode
            mask_jet = bh_fedd < bh_fedd_jet_max
            Lbol[mask_jet] = (0.1 * bh_Lbol_edd[mask_jet] * (bh_fedd[mask_jet]*10)**2).in_units('erg/s')
            group[f'bh_Lbol_acc_split'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {halo_type} bh_Lbol_acc_split: {error}\n')

        
        
        
        ## BH Luminosities ##########

        for Lbol_type, Lbol_name in zip(
            ['bh_Lbol', 'bh_Lbol_acc', 'bh_Lbol_acc_split'],
            ['Lbol', 'Lbol_acc', 'Lbol_acc_split']
        ):

            ## Florez+2021: Comparing simulations to AGN at z=0.75-2.25 with & without high X-ray luminosity (Lx>10^44 erg/s) AGN ###################################################################################
            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 3, 170 X-ray selected Type I AGN with Mbh available, OLS bisector Row
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Eddington ratio (fedd)
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}', shape=(group[f'bh_fedd_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}'].resize((group[f'bh_fedd_acc'].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
                slope = 0.752
                intercept = 2.134
                fedd_min = unyt.unyt_quantity(10**(-intercept/slope), '1')
                y = slope*np.log10(np.fmax(fedd, fedd_min)) + intercept
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type II AGN (low Lbol AGN)
                ## Table 4, 488 X-ray selected Type II AGN with Lbol & M* available, OLS bisector Row
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Eddington ratio (fedd)
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}', shape=(group[f'bh_fedd_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}'].resize((group[f'bh_fedd_acc'].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
                slope = 0.621
                intercept = 1.947
                fedd_min = unyt.unyt_quantity(10**(-intercept/slope), '1')
                y = slope*np.log10(np.fmax(fedd, fedd_min)) + intercept
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}: {error}\n')

            
            try:
                ## Soft (0.5-2 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.239
                a2 = 0.059
                a3 = -0.009
                b = 1.436
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')
            
            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.288
                a2 = 0.111
                a3 = -0.007
                b = 1.308
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) luminosity from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lband) with Lbol
                create_dataset(group, f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = -0.011
                a2 = -0.050
                a3 = 0.065
                b = 0.769
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')


            try:
                ## Soft (0.5-2 keV) Lx from Lusso+2012 for Type II AGN (high Lbol AGN)
                ## Table 2, 488 X-ray selected Type II AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.217
                a2 = 0.009
                a3 = -0.010
                b = 1.399
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type II AGN (high Lbol AGN)
                ## Table 2, 488 X-ray selected Type II AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.230
                a2 = 0.050
                a3 = 0.001
                b = 1.256
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')



            ## Florez+2021: Comparing simulations to AGN at z=0.75-2.25 with & without high X-ray luminosity (Lx>10^44 erg/s) AGN ###################################################################################
            try:
                ## Soft (0.5-2 keV) Lx from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lx_0.5-2keV_from_Lbol_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_from_Lbol_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 17.87
                k1 = 0.28
                c2 = 10.03
                k2 = -0.020
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_from_Lbol_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_0.5-2keV_from_Lbol_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lx_2-10keV_from_Lbol_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_from_Lbol_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 10.83
                k1 = 0.28
                c2 = 6.08
                k2 = -0.020
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_from_Lbol_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_from_Lbol_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) Luminosity from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_LB_0.44um_from_Lbol_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_from_Lbol_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 6.25
                k1 = -0.37
                c2 = 9.00
                k2 = -0.012
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_from_Lbol_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_from_Lbol_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## Mid-IR (15 um) Luminosity from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lmir_15um_from_Lbol_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lmir_15um_from_Lbol_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 7.40
                k1 = -0.37
                c2 = 10.66
                k2 = -0.014
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lmir_15um_from_Lbol_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lmir_15um_from_Lbol_Hopkins2007-{Lbol_name}: {error}\n')

            

            ############################################################################################################
            ## Hirschmann+2014: AGN luminosity functions and downsizing from cosmological simulations
            ## Comparing simulations to observed AGN luminosity functions in different bands
            ## using bolometric corrections from Marconi+2004
            try:
                ## Soft (0.5-2 keV) Lx
                create_dataset(group, f'bh_Lx_0.5-2keV_from_Lbol_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_from_Lbol_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.22
                a2 = 0.012
                a3 = -0.0015
                b = 1.65
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_from_Lbol_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_0.5-2keV_from_Lbol_Marconi2004-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx
                create_dataset(group, f'bh_Lx_2-10keV_from_Lbol_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_from_Lbol_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.24
                a2 = 0.012
                a3 = -0.0015
                b = 1.54
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_from_Lbol_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_from_Lbol_Marconi2004-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) Luminosity
                create_dataset(group, f'bh_LB_0.44um_from_Lbol_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_from_Lbol_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = -0.067
                a2 = 0.017
                a3 = -0.0023
                b = 0.80
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_from_Lbol_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_from_Lbol_Marconi2004-{Lbol_name}: {error}\n')

            

            ## Duras+2020 ###################################################################################
            try:
                ## 2-10 keV Lx for Type I AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 12.76
                b = 12.15
                c = 18.78
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for Type II AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 10.85
                b = 11.90
                c = 19.93
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 10.96
                b = 11.93
                c = 17.79
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}: {error}\n')


            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.13
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            
            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Eddington ratio
                create_dataset(group, f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group[f'bh_fedd_acc'], group[f'bh_fedd_acc'].attrs['units'])
                x = fedd
                a = 7.51
                b = 0.05
                c = 0.61
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Mbh
                create_dataset(group, f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                Mbh = unyt.unyt_array(group[f'bh_mass'], group[f'bh_mass'].attrs['units'])
                x = np.log10(Mbh.in_units('Msun'))
                a = 16.75
                b = 9.22
                c = 26.14
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}: {error}\n')

            
            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from eddington ratio
                create_dataset(group, f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.10
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from Mbh
                create_dataset(group, f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.05
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}: {error}\n')
        
            



        
        
        ## State of central SMBH
        try:
            create_dataset(group, f'nbh_no_accretion', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_no_accretion'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_filter = bh_fedd == unyt.unyt_quantity(0, '')
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_no_accretion'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_no_accretion: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            # bh_filter = np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= bh_mass_jet_min)
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            bh_quasar_low_fedd_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_filter = np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_filt)
            nbh = unyt.unyt_array(bh_quasar_filter.astype(int), '')
            group['nbh_quasar'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            # bh_filter = np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= a*bh_mass_jet_min)
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            bh_quasar_low_fedd_ascale_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_filter = np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_ascale_filt)
            nbh = unyt.unyt_array(bh_quasar_filter.astype(int), '')
            group['nbh_quasar_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_high_fedd', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_high_fedd'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            # bh_filter = bh_fedd >= bh_fedd_jet_max
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            nbh = unyt.unyt_array(bh_quasar_high_fedd_filt.astype(int), '')
            group['nbh_quasar_high_fedd'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_high_fedd: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_low_fedd', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_low_fedd'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            # bh_filter = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_low_fedd_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_low_fedd_filt.astype(int), '')
            group['nbh_quasar_low_fedd'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_low_fedd: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_low_fedd_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_low_fedd_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            # bh_filter = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_low_fedd_ascale_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_low_fedd_ascale_filt.astype(int), '')
            group['nbh_quasar_low_fedd_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_low_fedd_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.02', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.02'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_quasar_fedd002_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd002_filt.astype(int), '')
            group['nbh_quasar_fedd<0.02'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_fedd<0.02: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.02_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.02_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_quasar_fedd002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd002_ascale_filt.astype(int), '')
            group['nbh_quasar_fedd<0.02_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_fedd<0.02_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.002', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.002'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_quasar_fedd0002_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd0002_filt.astype(int), '')
            group['nbh_quasar_fedd<0.002'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_fedd<0.002: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.002_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.002_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_quasar_fedd0002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd0002_ascale_filt.astype(int), '')
            group['nbh_quasar_fedd<0.002_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_quasar_fedd<0.002_ascale: {error}\n')
        try:
            create_dataset(group, f'nbh_jet', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_jet'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_filter = np.logical_and(np.logical_and(bh_mass > bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_jet'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_jet: {error}\n')

        try:
            create_dataset(group, f'nbh_jet_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_jet_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_filter = np.logical_and(np.logical_and(bh_mass > a*bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_jet_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {halo_type} nbh_jet_ascale: {error}\n')




        for mass_type in halo_mass_types:
            try:
                create_dataset(group, f'f_{mass_type}-total', shape=(group[f'{mass_type}_mass'].shape[0],), dtype='f8', units='1')
                group[f'f_{mass_type}-total'].resize((group[f'{mass_type}_mass'].shape[0],))
                mass = np.nan_to_num(group[f'{mass_type}_mass'][:], nan=0.0, copy=True)
                mass = unyt.unyt_array(mass, group[f'{mass_type}_mass'].attrs['units'])
                total_mass = np.nan_to_num(group['total_mass'][:], nan=0.0, copy=True)
                total_mass = unyt.unyt_array(total_mass, group['total_mass'].attrs['units'])
                fraction = mass / total_mass
                group[f'f_{mass_type}-total'][:] = fraction.in_units('1')
            except Exception as error:
                print(f'\nError calculating {halo_type} f_{mass_type}-total: {error}\n')

            try:
                create_dataset(group, f'f_{mass_type}-gas', shape=(group[f'{mass_type}_mass'].shape[0],), dtype='f8', units='1')
                group[f'f_{mass_type}-gas'].resize((group[f'{mass_type}_mass'].shape[0],))
                mass = np.nan_to_num(group[f'{mass_type}_mass'][:], nan=0.0, copy=True)
                mass = unyt.unyt_array(mass, group[f'{mass_type}_mass'].attrs['units'])
                total_mass = np.nan_to_num(group['gas_mass'][:], nan=0.0, copy=True)
                total_mass = unyt.unyt_array(total_mass, group['gas_mass'].attrs['units'])
                fraction = mass / total_mass
                group[f'f_{mass_type}-gas'][:] = fraction.in_units('1')
            except Exception as error:
                print(f'\nError calculating {halo_type} f_{mass_type}-gas: {error}\n')




        try:
            create_dataset(group, 'sfr_snap', shape=(group['stellar_mass'].shape[0],), dtype='f8', units='Msun/yr')
            group['sfr_snap'].resize((group['stellar_mass'].shape[0],))
            values_curr = np.nan_to_num(group['stellar_mass'][:], nan=0.0, copy=True)
            values_prev = np.nan_to_num(np.roll(group['stellar_mass'][:], 1), nan=0.0, copy=True)
            values_prev[0] = 0.0
            mass_curr = unyt.unyt_array(values_curr, group['stellar_mass'].attrs['units'])
            mass_prev = unyt.unyt_array(values_prev, group['stellar_mass'].attrs['units'])
            delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
            sfr_snap = (mass_curr - mass_prev) / delta_t
            group['sfr_snap'][:] = sfr_snap.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {halo_type} sfr_snap: {error}\n')

        try:
            create_dataset(group, 'ssfr_snap', shape=(group['sfr_snap'].shape[0],), dtype='f8', units='yr**-1')
            group['ssfr_snap'].resize((group['sfr_snap'].shape[0],))
            sfr = np.nan_to_num(group['sfr_snap'][:], nan=0.0, copy=True)
            sfr = unyt.unyt_array(sfr, group['sfr_snap'].attrs['units'])
            stellar_mass = np.nan_to_num(group['stellar_mass'][:], nan=0.0, copy=True)
            stellar_mass = unyt.unyt_array(stellar_mass, group['stellar_mass'].attrs['units'])
            ssfr = sfr / stellar_mass
            group['ssfr_snap'][:] = ssfr.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {halo_type} ssfr_snap: {error}\n')

        try:
            create_dataset(group, 'stellar_mass_gradient', shape=(group['stellar_mass'].shape[0],), dtype='f8', units='Msun/yr')
            group['stellar_mass_gradient'].resize((group['stellar_mass'].shape[0],))
            values = np.nan_to_num(group['stellar_mass'][:], nan=0.0, copy=True)
            mass = unyt.unyt_array(values, group['stellar_mass'].attrs['units'])
            time = unyt.unyt_array(group['age'], group['age'].attrs['units'])
            gradient = np.gradient(mass.value, time.value, edge_order=1)
            gradient = unyt.unyt_array(gradient, mass.units/time.units)
            group['stellar_mass_gradient'][:] = gradient.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {halo_type} stellar_mass_gradient: {error}\n')

        try:
            create_dataset(group, 'stellar_mass_gradient_over_mstar', shape=(group['stellar_mass_gradient'].shape[0],), dtype='f8', units='yr**-1')
            group['stellar_mass_gradient_over_mstar'].resize((group['stellar_mass_gradient'].shape[0],))
            sfr = np.nan_to_num(group['stellar_mass_gradient'][:], nan=0.0, copy=True)
            sfr = unyt.unyt_array(sfr, group['stellar_mass_gradient'].attrs['units'])
            stellar_mass = np.nan_to_num(group['stellar_mass'][:], nan=0.0, copy=True)
            stellar_mass = unyt.unyt_array(stellar_mass, group['stellar_mass'].attrs['units'])
            ssfr = sfr / stellar_mass
            group['stellar_mass_gradient_over_mstar'][:] = ssfr.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {halo_type} stellar_mass_gradient_over_mstar: {error}\n')





        for aperture_name in halo_aperture_names:
            print(f'\n\nCalculating properties within aperture: {aperture_name}\n\n')

            ## Total mass in aperture
            try:
                create_dataset(group, f'm_total_{aperture_name}', shape=(group[f'mPartType0_{aperture_name}'].shape[0],), dtype='f8', units='Msun')
                group[f'm_total_{aperture_name}'].resize((group[f'mPartType0_{aperture_name}'].shape[0],))
                # total_mass = unyt.unyt_array(0.0, 'Msun')
                for part_type in part_types:
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    if part_type == part_types[0]:
                        total_mass = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                    else:
                        total_mass += unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                group[f'm_total_{aperture_name}'][:] = total_mass.in_units('Msun')
            except Exception as error:
                print(f'\nError calculating {halo_type} m_total_{aperture_name}: {error}\n')
            
            ## Mass fractions
            for part_type in part_types:
                try:
                    create_dataset(group, f'f_{part_type}_{aperture_name}-total', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                    group[f'f_{part_type}_{aperture_name}-total'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                    group[f'f_{part_type}_{aperture_name}-total'][:] = mass_fraction.in_units('1')
                except Exception as error:
                    print(f'\nError calculating {halo_type} f_{part_type}_{aperture_name}-total: {error}\n')

                try:
                    create_dataset(group, f'f_{part_type}_{aperture_name}-gas', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                    group[f'f_{part_type}_{aperture_name}-gas'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                    group[f'f_{part_type}_{aperture_name}-gas'][:] = mass_fraction.in_units('1')
                except Exception as error:
                    print(f'\nError calculating {halo_type} f_{part_type}_{aperture_name}-gas: {error}\n')


                if part_type == 'PartType0':
                    ## Very approximate number of new wind launches
                    try:
                        create_dataset(group, f'{part_type}_NNewWindLaunches_{aperture_name}', shape=(group[f'{part_type}_NWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'{part_type}_NNewWindLaunches_{aperture_name}'].resize((group[f'{part_type}_NWindLaunches_{aperture_name}'].shape[0],))
                        values_curr = np.nan_to_num(group[f'{part_type}_NWindLaunches_{aperture_name}'][:], nan=0.0, copy=True)
                        values_prev = np.nan_to_num(np.roll(group[f'{part_type}_NWindLaunches_{aperture_name}'][:], 1), nan=0.0, copy=True)
                        values_prev[0] = 0.0
                        n_curr = unyt.unyt_array(values_curr, group[f'{part_type}_NWindLaunches_{aperture_name}'].attrs['units'])
                        n_prev = unyt.unyt_array(values_prev, group[f'{part_type}_NWindLaunches_{aperture_name}'].attrs['units'])
                        n_new = n_curr - n_prev
                        group[f'{part_type}_NNewWindLaunches_{aperture_name}'][:] = n_new.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} NNewWindLaunches_{aperture_name}: {error}\n')

                    ## Very approximate gas mass in new wind launches
                    try:
                        create_dataset(group, f'm{part_type}_NewWindLaunches_{aperture_name}', shape=(group[f'{part_type}_NNewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='Msun')
                        group[f'm{part_type}_NewWindLaunches_{aperture_name}'].resize((group[f'{part_type}_NNewWindLaunches_{aperture_name}'].shape[0],))
                        n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                        m_new = n_new * mgas
                        group[f'm{part_type}_NewWindLaunches_{aperture_name}'][:] = m_new.in_units('Msun')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} m{part_type}_NewWindLaunches_{aperture_name}: {error}\n')

                    # ## Approximate mass loading factor within aperture
                    # try:
                    #     create_dataset(group, f'mass_loading_factor_{aperture_name}', shape=(group[f'sfr'].shape[0],), dtype='f8', units='1')
                    #     group[f'mass_loading_factor_{aperture_name}'].resize((group[f'sfr'].shape[0],))
                    #     n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                    #     m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                    #     sfr = unyt.unyt_array(group['sfr'], group['sfr'].attrs['units'])
                    #     mass_loading_factor = m_new / sfr
                    #     group[f'mass_loading_factor_{aperture_name}'][:] = mass_loading_factor.in_units('1')
                    # except Exception as error:
                    #     print(f'\nError calculating {halo_type} mass_loading_factor_{aperture_name}: {error}\n')

                    ## Approximate wind launch rate within aperture
                    try:
                        create_dataset(group, f'wind_launch_rate_{aperture_name}', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'wind_launch_rate_{aperture_name}'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        # n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
                        wind_launch_rate = m_new / delta_t
                        group[f'wind_launch_rate_{aperture_name}'][:] = wind_launch_rate.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} wind_launch_rate_{aperture_name}: {error}\n')

                    ## Approximate wind mass fraction within aperture
                    try:
                        create_dataset(group, f'f_wind_{aperture_name}-total', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_wind_{aperture_name}-total'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        total_mass = unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                        wind_mass_fraction = m_new / total_mass
                        group[f'f_wind_{aperture_name}-total'][:] = wind_mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} f_wind_{aperture_name}-total: {error}\n')

                    ## Approximate wind mass fraction of gas within aperture
                    try:
                        create_dataset(group, f'f_wind_{aperture_name}-gas', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_wind_{aperture_name}-gas'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        gas_mass = unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                        wind_mass_fraction_gas = m_new / gas_mass
                        group[f'f_wind_{aperture_name}-gas'][:] = wind_mass_fraction_gas.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} f_wind_{aperture_name}-gas: {error}\n')


                    ## Continue with other gas phases
                    for phase_def_name, phase_defs in gas_phases.items():
                        print(f'\nCalculating gas phases for definition: {phase_def_name}\n')
                        for phase_name in phase_defs:
                            print(f'Phase: {phase_name}')

                            ## Total mass fraction of gas phase within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm_total_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm_total_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {halo_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total: {error}\n')

                            ## Mass fraction of gas phase within aperture to total gas mass within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {halo_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas: {error}\n')

                            ## Mass fraction of gas phase within aperture to total coupled gas mass within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm{part_type}_Aviv_coupled_gas_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm{part_type}_Aviv_coupled_gas_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {halo_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas: {error}\n')


                    
                    ## Instantaneous star formation rate calculations
                    try:
                        create_dataset(group, f'ssfr_{part_type}_{aperture_name}', shape=(group[f'sfr_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_{part_type}_{aperture_name}'].resize((group[f'sfr_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'mPartType4_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'mPartType4_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} ssfr_{part_type}_{aperture_name}: {error}\n')

                    


                if part_type == 'PartType4':
                    ## Time-averaged star formation rate calculations
                    try:
                        create_dataset(group, f'ssfr_100Myr_{part_type}_{aperture_name}', shape=(group[f'sfr_100Myr_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_100Myr_{part_type}_{aperture_name}'].resize((group[f'sfr_100Myr_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_100Myr_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_100Myr_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'mPartType4_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'mPartType4_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_100Myr_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} ssfr_100Myr_{part_type}_{aperture_name}: {error}\n')



                    try:
                        create_dataset(group, f'sfr_snap_{part_type}_{aperture_name}', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'sfr_snap_{part_type}_{aperture_name}'].resize((group['stellar_mass'].shape[0],))
                        values_curr = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        values_prev = np.nan_to_num(np.roll(group[f'm{part_type}_{aperture_name}'][:], 1), nan=0.0, copy=True)
                        values_prev[0] = 0.0
                        mass_curr = unyt.unyt_array(values_curr, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        mass_prev = unyt.unyt_array(values_prev, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
                        sfr_snap = (mass_curr - mass_prev) / delta_t
                        group[f'sfr_snap_{part_type}_{aperture_name}'][:] = sfr_snap.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} sfr_snap_{part_type}_{aperture_name}: {error}\n')

                    try:
                        create_dataset(group, f'ssfr_snap_{part_type}_{aperture_name}', shape=(group[f'sfr_snap_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_snap_{part_type}_{aperture_name}'].resize((group[f'sfr_snap_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_snap_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_snap_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_snap_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} ssfr_snap_{part_type}_{aperture_name}: {error}\n')

                    try:
                        create_dataset(group, f'm{part_type}_{aperture_name}_gradient', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'm{part_type}_{aperture_name}_gradient'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        mass = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        time = unyt.unyt_array(group['age'], group['age'].attrs['units'])
                        gradient = np.gradient(mass.value, time.value, edge_order=1)
                        gradient = unyt.unyt_array(gradient, mass.units/time.units)
                        group[f'm{part_type}_{aperture_name}_gradient'][:] = gradient.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} m{part_type}_{aperture_name}_gradient: {error}\n')

                    try:
                        create_dataset(group, f'm{part_type}_{aperture_name}_gradient_over_mstar', shape=(group[f'm{part_type}_{aperture_name}_gradient'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'm{part_type}_{aperture_name}_gradient_over_mstar'].resize((group[f'm{part_type}_{aperture_name}_gradient'].shape[0],))
                        sfr = np.nan_to_num(group[f'm{part_type}_{aperture_name}_gradient'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'm{part_type}_{aperture_name}_gradient'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'm{part_type}_{aperture_name}_gradient_over_mstar'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} m{part_type}_{aperture_name}_gradient_over_mstar: {error}\n')


                if part_type == 'PartType5':
                    try:
                        create_dataset(group, f'f_{part_type}_phys_{aperture_name}-total', shape=(group[f'm{part_type}_phys_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_{part_type}_phys_{aperture_name}-total'].resize((group[f'm{part_type}_phys_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_phys_{aperture_name}'][:], nan=0.0, copy=True)
                        mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_phys_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                        group[f'f_{part_type}_phys_{aperture_name}-total'][:] = mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} f_{part_type}_phys_{aperture_name}-total: {error}\n')

                    try:
                        create_dataset(group, f'f_{part_type}_phys_{aperture_name}-gas', shape=(group[f'm{part_type}_phys_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_{part_type}_phys_{aperture_name}-gas'].resize((group[f'm{part_type}_phys_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_phys_{aperture_name}'][:], nan=0.0, copy=True)
                        mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_phys_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                        group[f'f_{part_type}_phys_{aperture_name}-gas'][:] = mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {halo_type} f_{part_type}_phys_{aperture_name}-gas: {error}\n')





    
    print('\n\n\n\n\n\n')




    
    
    for central_type in args.central_types:
        print(f'\n\n\nCalculating properties for central type: {central_type}\n')
        try:
            group = f[f'/{central_type}']
        except:
            print(f'\n{central_type} does not exist!\n')
            continue

        ## Scale factor
        try:
            create_dataset(group, f'a', shape=(group[f'z'].shape[0],), dtype='f8', units=group[f'z'].attrs['units'])
            group[f'a'].resize((group[f'z'].shape[0],))
            z = unyt.unyt_array(group['z'], group[f'z'].attrs['units'])
            a = 1./(1. + z)
            group['a'][:] = a
        except Exception as error:
            print(f'\nError calculating {central_type} a: {error}\n')


        ## Time since last snapshot
        try:
            create_dataset(group, f'delta_t', shape=(group[f'age'].shape[0],), dtype='f8', units=group[f'age'].attrs['units'])
            group[f'delta_t'].resize((group[f'age'].shape[0],))
            values_curr = np.nan_to_num(group[f'age'][:], nan=0.0, copy=True)
            values_prev = np.nan_to_num(np.roll(group[f'age'][:], 1), nan=0.0, copy=True)
            values_prev[0] = 0.0
            age_curr = unyt.unyt_array(values_curr, group[f'age'].attrs['units'])
            age_prev = unyt.unyt_array(values_prev, group[f'age'].attrs['units'])
            delta_t = age_curr - age_prev
            group[f'delta_t'][:] = delta_t.in_units(group[f'age'].attrs['units'])
        except Exception as error:
            print(f'\nError calculating {central_type} delta_t: {error}\n')
        
        try:
            create_dataset(group, 'ssfr', shape=(group['sfr'].shape[0],), dtype='f8', units='yr**-1')
            group['ssfr'].resize((group['sfr'].shape[0],))
            ssfr = unyt.unyt_array(group['sfr'], group['sfr'].attrs['units'])/unyt.unyt_array(group['stellar_mass'], group['stellar_mass'].attrs['units'])
            group['ssfr'][:] = ssfr.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {central_type} ssfr: {error}\n')

        try:
            create_dataset(group, 'ssfr_100', shape=(group['sfr_100'].shape[0],), dtype='f8', units='yr**-1')
            group['ssfr_100'].resize((group['sfr_100'].shape[0],))
            ssfr_100 = unyt.unyt_array(group['sfr_100'], group['sfr_100'].attrs['units'])/unyt.unyt_array(group['stellar_mass'], group['stellar_mass'].attrs['units'])
            group['ssfr_100'][:] = ssfr_100.in_units('yr**-1')
        except Exception as error:
            print(f'\nError calculating {central_type} ssfr_100: {error}\n')




        ## BH accretion rates and bolometric luminosities
        try:
            create_dataset(group, 'bh_mdot_edd-bad', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_edd-bad'].resize((group['bh_mdot'].shape[0],))
            bh_mdot_edd = unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units'])/unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            group['bh_mdot_edd-bad'][:] = bh_mdot_edd.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_mdot_edd-bad: {error}\n')

        try:
            create_dataset(group, 'bh_mdot_edd', shape=(group['bh_mass'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_edd'].resize((group['bh_mass'].shape[0],))
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson))
            group['bh_mdot_edd'][:] = bh_mdot_edd.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_mdot_edd: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol', shape=(group[f'bh_mdot'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol'].resize((group[f'bh_mdot'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units'])
            group[f'bh_Lbol'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_Lbol: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol_edd', shape=(group[f'bh_mdot_edd'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_edd'].resize((group[f'bh_mdot_edd'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot_edd'], group['bh_mdot_edd'].attrs['units'])
            group[f'bh_Lbol_edd'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_Lbol_edd: {error}\n')

        try:
            create_dataset(group, 'bh_mdot_acc', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='Msun/yr')
            group['bh_mdot_acc'].resize((group[f'bh_mdot'].shape[0],))
            bh_mdot_acc = unyt.unyt_array(group['bh_mdot'], group['bh_mdot'].attrs['units']) / (1 - f_rad)
            group['bh_mdot_acc'][:] = bh_mdot_acc.in_units('Msun/yr')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_mdot_acc: {error}\n')

        try:
            create_dataset(group, f'bh_Lbol_acc', shape=(group[f'bh_mdot_acc'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_acc'].resize((group[f'bh_mdot_acc'].shape[0],))
            Lbol = f_rad * unyt.c**2 * unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            group[f'bh_Lbol_acc'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_Lbol_acc: {error}\n')
            # continue

        try:
            create_dataset(group, 'bh_fedd_acc', shape=(group['bh_mdot_acc'].shape[0],), dtype='f8', units='1')
            group['bh_fedd_acc'].resize((group[f'bh_mdot_acc'].shape[0],))
            bh_mdot = unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            bh_mdot_edd = unyt.unyt_array(group['bh_mdot_edd'], group['bh_mdot_edd'].attrs['units'])
            bh_fedd_acc = bh_mdot/bh_mdot_edd
            group['bh_fedd_acc'][:] = bh_fedd_acc.in_units('1')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_fedd_acc: {error}\n')

        
        try:
            ## From Hirschmann+2014, equation 6 & 7, originally from Churazov+2005
            create_dataset(group, f'bh_Lbol_acc_split', shape=(group[f'bh_mdot_acc'].shape[0],), dtype='f8', units='erg/s')
            group[f'bh_Lbol_acc_split'].resize((group[f'bh_mdot_acc'].shape[0],))
            bh_mdot = unyt.unyt_array(group['bh_mdot_acc'], group['bh_mdot_acc'].attrs['units'])
            bh_Lbol_edd = unyt.unyt_array(group['bh_Lbol_edd'], group['bh_Lbol_edd'].attrs['units'])
            bh_fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
            Lbol = unyt.unyt_array(np.zeros(bh_mdot.shape[0]), 'erg/s')
            # Radiative mode
            mask_rad = bh_fedd >= bh_fedd_jet_max
            Lbol[mask_rad] = ((f_rad/(1.-f_rad)) * bh_mdot[mask_rad] * unyt.c**2).in_units('erg/s')
            # Mechanical mode
            mask_jet = bh_fedd < bh_fedd_jet_max
            Lbol[mask_jet] = (0.1 * bh_Lbol_edd[mask_jet] * (bh_fedd[mask_jet]*10)**2).in_units('erg/s')
            group[f'bh_Lbol_acc_split'][:] = Lbol.in_units('erg/s')
        except Exception as error:
            print(f'\nError calculating {central_type} bh_Lbol_acc_split: {error}\n')

        




        ## BH Luminosities ##########

        for Lbol_type, Lbol_name in zip(
            ['bh_Lbol', 'bh_Lbol_acc', 'bh_Lbol_acc_split'],
            ['Lbol', 'Lbol_acc', 'Lbol_acc_split']
        ):

            ## Florez+2021: Comparing simulations to AGN at z=0.75-2.25 with & without high X-ray luminosity (Lx>10^44 erg/s) AGN
            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 3, 170 X-ray selected Type I AGN with Mbh available, OLS bisector Row
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Eddington ratio (fedd)
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}', shape=(group[f'bh_fedd_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}'].resize((group[f'bh_fedd_acc'].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
                slope = 0.752
                intercept = 2.134
                fedd_min = unyt.unyt_quantity(10**(-intercept/slope), '1')
                y = slope*np.log10(np.fmax(fedd, fedd_min)) + intercept
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_sLx_2-10keV_typeI_from_fedd_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type II AGN (low Lbol AGN)
                ## Table 4, 488 X-ray selected Type II AGN with Lbol & M* available, OLS bisector Row
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Eddington ratio (fedd)
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}', shape=(group[f'bh_fedd_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}'].resize((group[f'bh_fedd_acc'].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group['bh_fedd_acc'], group['bh_fedd_acc'].attrs['units'])
                slope = 0.621
                intercept = 1.947
                fedd_min = unyt.unyt_quantity(10**(-intercept/slope), '1')
                y = slope*np.log10(np.fmax(fedd, fedd_min)) + intercept
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_2-10keV_typeII_from_fedd_Lusso2012-{Lbol_name}: {error}\n')

            
            try:
                ## Soft (0.5-2 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.239
                a2 = 0.059
                a3 = -0.009
                b = 1.436
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_0.5-2keV_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')
            
            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.288
                a2 = 0.111
                a3 = -0.007
                b = 1.308
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_2-10keV_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) luminosity from Lusso+2012 for Type I AGN (high Lbol AGN)
                ## Table 2, 373 X-ray selected Type I AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lband) with Lbol
                create_dataset(group, f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[f'bh_Lbol_acc'].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = -0.011
                a2 = -0.050
                a3 = 0.065
                b = 0.769
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_LB_0.44um_typeI_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')


            try:
                ## Soft (0.5-2 keV) Lx from Lusso+2012 for Type II AGN (high Lbol AGN)
                ## Table 2, 488 X-ray selected Type II AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.217
                a2 = 0.009
                a3 = -0.010
                b = 1.399
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_0.5-2keV_typeII_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Lusso+2012 for Type II AGN (high Lbol AGN)
                ## Table 2, 488 X-ray selected Type II AGN with spectroscopy+photometry
                ## Calculated from scaling relation of bolometric correction (BC = Lbol/Lx) with Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.230
                a2 = 0.050
                a3 = 0.001
                b = 1.256
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_2-10keV_typeII_from_Lbol_Lusso2012-{Lbol_name}: {error}\n')



            ## Florez+2021: Comparing simulations to AGN at z=0.75-2.25 with & without high X-ray luminosity (Lx>10^44 erg/s) AGN
            try:
                ## Soft (0.5-2 keV) Lx from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lx_0.5-2keV_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 17.87
                k1 = 0.28
                c2 = 10.03
                k2 = -0.020
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_0.5-2keV_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lx_2-10keV_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 10.83
                k1 = 0.26
                c2 = 6.08
                k2 = -0.02
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_2-10keV_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) Luminosity from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_LB_0.44um_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 6.25
                k1 = -0.37
                c2 = 9.00
                k2 = -0.012
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_LB_0.44um_Hopkins2007-{Lbol_name}: {error}\n')

            try:
                ## Mid-IR (15 um) Luminosity from Hopkins+2007
                ## authors use fully integrated SEDs of quasars from hard X-rays to radio wavelengths,
                ## column densities for a given spectral shape, and X-ray luminosities
                ## to derive a relation between X-ray luminosity and bolometric luminosity
                create_dataset(group, f'bh_Lmir_15um_Hopkins2007-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lmir_15um_Hopkins2007-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = (Lbol/Lsun)*10**(-10.0)
                c1 = 7.40
                k1 = -0.37
                c2 = 10.66
                k2 = -0.014
                y = c1*x**k1 + c2*x**k2
                Lx = Lbol * 10**(-y)
                group[f'bh_Lmir_15um_Hopkins2007-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lmir_15um_Hopkins2007-{Lbol_name}: {error}\n')

            

            ## Hirschmann+2014: AGN luminosity functions and downsizing from cosmological simulations
            ## Comparing simulations to observed AGN luminosity functions in different bands
            ## using bolometric corrections from Marconi+2004
            try:
                ## Soft (0.5-2 keV) Lx
                create_dataset(group, f'bh_Lx_0.5-2keV_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_0.5-2keV_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.22
                a2 = 0.012
                a3 = -0.0015
                b = 1.65
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_0.5-2keV_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_0.5-2keV_Marconi2004-{Lbol_name}: {error}\n')

            try:
                ## Hard (2-10 keV) Lx
                create_dataset(group, f'bh_Lx_2-10keV_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = 0.24
                a2 = 0.012
                a3 = -0.0015
                b = 1.54
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_Lx_2-10keV_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_Lx_2-10keV_Marconi2004-{Lbol_name}: {error}\n')

            try:
                ## B band (0.44 um) Luminosity
                create_dataset(group, f'bh_LB_0.44um_Marconi2004-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_Marconi2004-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun) - 12.0
                a1 = -0.067
                a2 = 0.017
                a3 = -0.0023
                b = 0.80
                y = a1*x + a2*x**2 + a3*x**3 + b
                Lx = Lbol * 10**(-y)
                group[f'bh_LB_0.44um_Marconi2004-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {central_type} bh_LB_0.44um_Marconi2004-{Lbol_name}: {error}\n')

            

            ## Duras+2020 ###################################################################################
            try:
                ## 2-10 keV Lx for Type I AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 12.76
                b = 12.15
                c = 18.78
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeI_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for Type II AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 10.85
                b = 11.90
                c = 19.93
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_typeII_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                x = np.log10(Lbol/Lsun)
                a = 10.96
                b = 11.93
                c = 17.79
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_Lbol_Duras2020-{Lbol_name}: {error}\n')


            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from Lbol
                create_dataset(group, f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.13
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_Lbol_Duras2020-{Lbol_name}: {error}\n')

            
            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Eddington ratio
                create_dataset(group, f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                fedd = unyt.unyt_array(group[f'bh_fedd_acc'], group[f'bh_fedd_acc'].attrs['units'])
                x = fedd
                a = 7.51
                b = 0.05
                c = 0.61
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_fedd_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## 2-10 keV Lx for general AGN from Duras+2020, calculated from Mbh
                create_dataset(group, f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                Mbh = unyt.unyt_array(group[f'bh_mass'], group[f'bh_mass'].attrs['units'])
                x = np.log10(Mbh.in_units('Msun'))
                a = 16.75
                b = 9.22
                c = 26.14
                y = a*(1+(x/b)**c)
                Lx = Lbol/y
                group[f'bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}'][:] = Lx.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_Lx_2-10keV_general_from_Mbh_Duras2020-{Lbol_name}: {error}\n')

            
            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from eddington ratio
                create_dataset(group, f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.10
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_fedd_Duras2020-{Lbol_name}: {error}\n')

            try:
                ## Optical B-band (0.44 um) luminosity for general AGN from Duras+2020, calculated from Mbh
                create_dataset(group, f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}', shape=(group[Lbol_type].shape[0],), dtype='f8', units='erg/s')
                group[f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}'].resize((group[Lbol_type].shape[0],))
                Lbol = unyt.unyt_array(group[Lbol_type], group[Lbol_type].attrs['units'])
                # x = np.log10(Lbol/Lsun)
                # a = 10.96
                # b = 11.93
                # c = 17.79
                y = 5.05
                L = Lbol/y
                group[f'bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}'][:] = L.in_units('erg/s')
            except Exception as error:
                print(f'\nError calculating {halo_type} bh_LB_0.44um_general_from_Mbh_Duras2020-{Lbol_name}: {error}\n')




        
        
        ## State of central SMBH
        try:
            create_dataset(group, f'nbh_no_accretion', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_no_accretion'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_filter = bh_fedd == unyt.unyt_quantity(0, '')
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_no_accretion'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_no_accretion: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            # bh_filter = np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= bh_mass_jet_min)
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            bh_quasar_low_fedd_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_filter = np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_filt)
            nbh = unyt.unyt_array(bh_quasar_filter.astype(int), '')
            group['nbh_quasar'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            # bh_filter = np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= a*bh_mass_jet_min)
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            bh_quasar_low_fedd_ascale_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_filter = np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_ascale_filt)
            nbh = unyt.unyt_array(bh_quasar_filter.astype(int), '')
            group['nbh_quasar_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_high_fedd', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_high_fedd'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            # bh_filter = bh_fedd >= bh_fedd_jet_max
            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
            nbh = unyt.unyt_array(bh_quasar_high_fedd_filt.astype(int), '')
            group['nbh_quasar_high_fedd'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_high_fedd: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_low_fedd', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_low_fedd'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            # bh_filter = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_low_fedd_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_low_fedd_filt.astype(int), '')
            group['nbh_quasar_low_fedd'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_low_fedd: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_low_fedd_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_low_fedd_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            # bh_filter = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            bh_quasar_low_fedd_ascale_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_low_fedd_ascale_filt.astype(int), '')
            group['nbh_quasar_low_fedd_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_low_fedd_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.02', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.02'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_quasar_fedd002_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd002_filt.astype(int), '')
            group['nbh_quasar_fedd<0.02'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_fedd<0.02: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.02_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.02_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_quasar_fedd002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd002_ascale_filt.astype(int), '')
            group['nbh_quasar_fedd<0.02_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_fedd<0.02_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.002', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.002'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_quasar_fedd0002_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd0002_filt.astype(int), '')
            group['nbh_quasar_fedd<0.002'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_fedd<0.002: {error}\n')

        try:
            create_dataset(group, f'nbh_quasar_fedd<0.002_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_quasar_fedd<0.002_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_quasar_fedd0002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= a*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_quasar_fedd0002_ascale_filt.astype(int), '')
            group['nbh_quasar_fedd<0.002_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_quasar_fedd<0.002_ascale: {error}\n')

        try:
            create_dataset(group, f'nbh_jet', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_jet'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            bh_filter = np.logical_and(np.logical_and(bh_mass > bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_jet'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_jet: {error}\n')

        try:
            create_dataset(group, f'nbh_jet_ascale', shape=(group[f'bh_fedd'].shape[0],), dtype='f8', units='1')
            group[f'nbh_jet_ascale'].resize((group[f'bh_fedd'].shape[0],))
            bh_fedd = unyt.unyt_array(group['bh_fedd'], group['bh_fedd'].attrs['units'])
            bh_mass = unyt.unyt_array(group['bh_mass'], group['bh_mass'].attrs['units'])
            a = unyt.unyt_array(group['a'], group['a'].attrs['units'])
            bh_filter = np.logical_and(np.logical_and(bh_mass > a*bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, ''))
            nbh = unyt.unyt_array(bh_filter.astype(int), '')
            group['nbh_jet_ascale'][:] = nbh
        except Exception as error:
            print(f'\nError calculating {central_type} nbh_jet_ascale: {error}\n')

            






        for total_mass_type, mass_subtypes in zip(central_mass_total_types, central_mass_subtypes):
            for mass_subtype in mass_subtypes:
                try:
                    create_dataset(group, f'f_{mass_subtype}-{total_mass_type}', shape=(group[f'{mass_subtype}_mass'].shape[0],), dtype='f8', units='1')
                    group[f'f_{mass_subtype}-{total_mass_type}'].resize((group[f'{mass_subtype}_mass'].shape[0],))
                    mass = np.nan_to_num(group[f'{mass_subtype}_mass'][:], nan=0.0, copy=True)
                    mass = unyt.unyt_array(mass, group[f'{mass_subtype}_mass'].attrs['units'])
                    total_mass = np.nan_to_num(group[f'{total_mass_type}_mass'][:], nan=0.0, copy=True)
                    total_mass = unyt.unyt_array(total_mass, group[f'{total_mass_type}_mass'].attrs['units'])
                    mass_ratio = mass / total_mass
                    group[f'f_{mass_subtype}-{total_mass_type}'][:] = mass_ratio.in_units('1')
                except Exception as error:
                    print(f'\nError calculating {central_type} f_{mass_subtype}-{total_mass_type}: {error}\n')





        for star_mass_type, star_mass_name in zip(central_star_masses_for_sfr, central_star_names_for_sfr):
        
            try:
                create_dataset(group, f'sfr_snap{star_mass_name}', shape=(group[f'{star_mass_type}_mass'].shape[0],), dtype='f8', units='Msun/yr')
                group[f'sfr_snap{star_mass_name}'].resize((group[f'{star_mass_type}_mass'].shape[0],))
                values_curr = np.nan_to_num(group[f'{star_mass_type}_mass'][:], nan=0.0, copy=True)
                values_prev = np.nan_to_num(np.roll(group[f'{star_mass_type}_mass'][:], 1), nan=0.0, copy=True)
                values_prev[0] = 0.0
                mass_curr = unyt.unyt_array(values_curr, group[f'{star_mass_type}_mass'].attrs['units'])
                mass_prev = unyt.unyt_array(values_prev, group[f'{star_mass_type}_mass'].attrs['units'])
                delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
                sfr_snap = (mass_curr - mass_prev) / delta_t
                group[f'sfr_snap{star_mass_name}'][:] = sfr_snap.in_units('Msun/yr')
            except Exception as error:
                print(f'\nError calculating {central_type} sfr_snap{star_mass_name}: {error}\n')

            try:
                create_dataset(group, f'ssfr_snap{star_mass_name}', shape=(group[f'sfr_snap{star_mass_name}'].shape[0],), dtype='f8', units='yr**-1')
                group[f'ssfr_snap{star_mass_name}'].resize((group[f'sfr_snap{star_mass_name}'].shape[0],))
                sfr = np.nan_to_num(group[f'sfr_snap{star_mass_name}'][:], nan=0.0, copy=True)
                sfr = unyt.unyt_array(sfr, group[f'sfr_snap{star_mass_name}'].attrs['units'])
                stellar_mass = np.nan_to_num(group[f'{star_mass_type}_mass'][:], nan=0.0, copy=True)
                stellar_mass = unyt.unyt_array(stellar_mass, group[f'{star_mass_type}_mass'].attrs['units'])
                ssfr = sfr / stellar_mass
                group[f'ssfr_snap{star_mass_name}'][:] = ssfr.in_units('yr**-1')
            except Exception as error:
                print(f'\nError calculating {central_type} ssfr_snap{star_mass_name}: {error}\n')

            try:
                create_dataset(group, f'{star_mass_type}_mass_gradient', shape=(group[f'{star_mass_type}_mass'].shape[0],), dtype='f8', units='Msun/yr')
                group[f'{star_mass_type}_mass_gradient'].resize((group[f'{star_mass_type}_mass'].shape[0],))
                values = np.nan_to_num(group[f'{star_mass_type}_mass'][:], nan=0.0, copy=True)
                mass = unyt.unyt_array(values, group[f'{star_mass_type}_mass'].attrs['units'])
                time = unyt.unyt_array(group['age'], group['age'].attrs['units'])
                gradient = np.gradient(mass.value, time.value, edge_order=1)
                gradient = unyt.unyt_array(gradient, mass.units/time.units)
                group[f'{star_mass_type}_mass_gradient'][:] = gradient.in_units('Msun/yr')
            except Exception as error:
                print(f'\nError calculating {central_type} {star_mass_type}_mass_gradient: {error}\n')

            try:
                create_dataset(group, f'{star_mass_type}_mass_gradient_over_mstar', shape=(group[f'{star_mass_type}_mass_gradient'].shape[0],), dtype='f8', units='yr**-1')
                group[f'{star_mass_type}_mass_gradient_over_mstar'].resize((group[f'{star_mass_type}_mass_gradient'].shape[0],))
                sfr = np.nan_to_num(group[f'{star_mass_type}_mass_gradient'][:], nan=0.0, copy=True)
                sfr = unyt.unyt_array(sfr, group[f'{star_mass_type}_mass_gradient'].attrs['units'])
                stellar_mass = np.nan_to_num(group[f'{star_mass_type}_mass'][:], nan=0.0, copy=True)
                stellar_mass = unyt.unyt_array(stellar_mass, group[f'{star_mass_type}_mass'].attrs['units'])
                ssfr = sfr / stellar_mass
                group[f'{star_mass_type}_mass_gradient_over_mstar'][:] = ssfr.in_units('yr**-1')
            except Exception as error:
                print(f'\nError calculating {central_type} {star_mass_type}_mass_gradient_over_mstar: {error}\n')




        for aperture_name in central_aperture_names:
            print(f'\n\nCalculating properties within aperture: {aperture_name}\n\n')

            ## Total mass in aperture
            try:
                create_dataset(group, f'm_total_{aperture_name}', shape=(group[f'mPartType0_{aperture_name}'].shape[0],), dtype='f8', units='Msun')
                group[f'm_total_{aperture_name}'].resize((group[f'mPartType0_{aperture_name}'].shape[0],))
                # total_mass = unyt.unyt_array(0.0, 'Msun')
                for part_type in part_types:
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    if part_type == part_types[0]:
                        total_mass = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                    else:
                        total_mass += unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                group[f'm_total_{aperture_name}'][:] = total_mass.in_units('Msun')
            except Exception as error:
                print(f'\nError calculating {central_type} m_total_{aperture_name}: {error}\n')
            
            ## Mass fractions
            for part_type in part_types:
                try:
                    create_dataset(group, f'f_{part_type}_{aperture_name}-total', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                    group[f'f_{part_type}_{aperture_name}-total'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                    group[f'f_{part_type}_{aperture_name}-total'][:] = mass_fraction.in_units('1')
                except Exception as error:
                    print(f'\nError calculating {central_type} f_{part_type}_{aperture_name}-total: {error}\n')

                try:
                    create_dataset(group, f'f_{part_type}_{aperture_name}-gas', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                    group[f'f_{part_type}_{aperture_name}-gas'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                    values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                    mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                    group[f'f_{part_type}_{aperture_name}-gas'][:] = mass_fraction.in_units('1')
                except Exception as error:
                    print(f'\nError calculating {central_type} f_{part_type}_{aperture_name}-gas: {error}\n')


                if part_type == 'PartType0':
                    ## Very approximate number of new wind launches
                    try:
                        create_dataset(group, f'{part_type}_NNewWindLaunches_{aperture_name}', shape=(group[f'{part_type}_NWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'{part_type}_NNewWindLaunches_{aperture_name}'].resize((group[f'{part_type}_NWindLaunches_{aperture_name}'].shape[0],))
                        values_curr = np.nan_to_num(group[f'{part_type}_NWindLaunches_{aperture_name}'][:], nan=0.0, copy=True)
                        values_prev = np.nan_to_num(np.roll(group[f'{part_type}_NWindLaunches_{aperture_name}'][:], 1), nan=0.0, copy=True)
                        values_prev[0] = 0.0
                        n_curr = unyt.unyt_array(values_curr, group[f'{part_type}_NWindLaunches_{aperture_name}'].attrs['units'])
                        n_prev = unyt.unyt_array(values_prev, group[f'{part_type}_NWindLaunches_{aperture_name}'].attrs['units'])
                        n_new = n_curr - n_prev
                        group[f'{part_type}_NNewWindLaunches_{aperture_name}'][:] = n_new.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} NNewWindLaunches_{aperture_name}: {error}\n')

                    ## Very approximate gas mass in new wind launches
                    try:
                        create_dataset(group, f'm{part_type}_NewWindLaunches_{aperture_name}', shape=(group[f'{part_type}_NNewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='Msun')
                        group[f'm{part_type}_NewWindLaunches_{aperture_name}'].resize((group[f'{part_type}_NNewWindLaunches_{aperture_name}'].shape[0],))
                        n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                        m_new = n_new * mgas
                        group[f'm{part_type}_NewWindLaunches_{aperture_name}'][:] = m_new.in_units('Msun')
                    except Exception as error:
                        print(f'\nError calculating {central_type} m{part_type}_NewWindLaunches_{aperture_name}: {error}\n')

                    # ## Approximate mass loading factor within aperture
                    # try:
                    #     create_dataset(group, f'mass_loading_factor_{aperture_name}', shape=(group[f'sfr'].shape[0],), dtype='f8', units='1')
                    #     group[f'mass_loading_factor_{aperture_name}'].resize((group[f'sfr'].shape[0],))
                    #     n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                    #     m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                    #     sfr = unyt.unyt_array(group['sfr'], group['sfr'].attrs['units'])
                    #     mass_loading_factor = m_new / sfr
                    #     group[f'mass_loading_factor_{aperture_name}'][:] = mass_loading_factor.in_units('1')
                    # except Exception as error:
                    #     print(f'\nError calculating {central_type} mass_loading_factor_{aperture_name}: {error}\n')

                    ## Approximate wind launch rate within aperture
                    try:
                        create_dataset(group, f'wind_launch_rate_{aperture_name}', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'wind_launch_rate_{aperture_name}'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        # n_new = unyt.unyt_array(group[f'{part_type}_NNewWindLaunches_{aperture_name}'], group[f'{part_type}_NNewWindLaunches_{aperture_name}'].attrs['units'])
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
                        wind_launch_rate = m_new / delta_t
                        group[f'wind_launch_rate_{aperture_name}'][:] = wind_launch_rate.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {central_type} wind_launch_rate_{aperture_name}: {error}\n')

                    ## Approximate wind mass fraction within aperture
                    try:
                        create_dataset(group, f'f_wind_{aperture_name}-total', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_wind_{aperture_name}-total'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        total_mass = unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                        wind_mass_fraction = m_new / total_mass
                        group[f'f_wind_{aperture_name}-total'][:] = wind_mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} f_wind_{aperture_name}-total: {error}\n')

                    ## Approximate wind mass fraction of gas within aperture
                    try:
                        create_dataset(group, f'f_wind_{aperture_name}-gas', shape=(group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_wind_{aperture_name}-gas'].resize((group[f'm{part_type}_NewWindLaunches_{aperture_name}'].shape[0],))
                        m_new = unyt.unyt_array(group[f'm{part_type}_NewWindLaunches_{aperture_name}'], group[f'm{part_type}_NewWindLaunches_{aperture_name}'].attrs['units'])
                        gas_mass = unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                        wind_mass_fraction_gas = m_new / gas_mass
                        group[f'f_wind_{aperture_name}-gas'][:] = wind_mass_fraction_gas.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} f_wind_{aperture_name}-gas: {error}\n')


                    ## Continue with other gas phases
                    for phase_def_name, phase_defs in gas_phases.items():
                        print(f'\nCalculating gas phases for definition: {phase_def_name}\n')
                        for phase_name in phase_defs:
                            print(f'Phase: {phase_name}')

                            ## Total mass fraction of gas phase within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm_total_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm_total_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {central_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-total: {error}\n')

                            ## Mass fraction of gas phase within aperture to total gas mass within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {central_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-gas: {error}\n')

                            ## Mass fraction of gas phase within aperture to total coupled gas mass within aperture
                            try:
                                create_dataset(group, f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas', shape=(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],), dtype='f8', units='1')
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas'].resize((group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].shape[0],))
                                values_phase = np.nan_to_num(group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_phase = unyt.unyt_array(values_phase, group[f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'].attrs['units'])
                                values_total = np.nan_to_num(group[f'm{part_type}_Aviv_coupled_gas_{aperture_name}'][:], nan=0.0, copy=True)
                                mass_total = unyt.unyt_array(values_total, group[f'm{part_type}_Aviv_coupled_gas_{aperture_name}'].attrs['units'])
                                phase_mass_fraction = mass_phase / mass_total
                                group[f'f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas'][:] = phase_mass_fraction.in_units('1')
                            except Exception as error:
                                print(f'\nError calculating {central_type} f_{part_type}_{phase_def_name}_{phase_name}_{aperture_name}-coupled_gas: {error}\n')


                    
                    ## Instantaneous star formation rate calculations
                    try:
                        create_dataset(group, f'ssfr_{part_type}_{aperture_name}', shape=(group[f'sfr_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_{part_type}_{aperture_name}'].resize((group[f'sfr_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'mPartType4_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'mPartType4_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} ssfr_{part_type}_{aperture_name}: {error}\n')

                    


                if part_type == 'PartType4':
                    ## Time-averaged star formation rate calculations
                    try:
                        create_dataset(group, f'ssfr_100Myr_{part_type}_{aperture_name}', shape=(group[f'sfr_100Myr_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_100Myr_{part_type}_{aperture_name}'].resize((group[f'sfr_100Myr_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_100Myr_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_100Myr_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'mPartType4_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'mPartType4_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_100Myr_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} ssfr_100Myr_{part_type}_{aperture_name}: {error}\n')



                    try:
                        create_dataset(group, f'sfr_snap_{part_type}_{aperture_name}', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'sfr_snap_{part_type}_{aperture_name}'].resize((group['stellar_mass'].shape[0],))
                        values_curr = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        values_prev = np.nan_to_num(np.roll(group[f'm{part_type}_{aperture_name}'][:], 1), nan=0.0, copy=True)
                        values_prev[0] = 0.0
                        mass_curr = unyt.unyt_array(values_curr, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        mass_prev = unyt.unyt_array(values_prev, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        delta_t = unyt.unyt_array(group['delta_t'], group['delta_t'].attrs['units'])
                        sfr_snap = (mass_curr - mass_prev) / delta_t
                        group[f'sfr_snap_{part_type}_{aperture_name}'][:] = sfr_snap.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {central_type} sfr_snap_{part_type}_{aperture_name}: {error}\n')

                    try:
                        create_dataset(group, f'ssfr_snap_{part_type}_{aperture_name}', shape=(group[f'sfr_snap_{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'ssfr_snap_{part_type}_{aperture_name}'].resize((group[f'sfr_snap_{part_type}_{aperture_name}'].shape[0],))
                        sfr = np.nan_to_num(group[f'sfr_snap_{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'sfr_snap_{part_type}_{aperture_name}'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'ssfr_snap_{part_type}_{aperture_name}'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} ssfr_snap_{part_type}_{aperture_name}: {error}\n')

                    try:
                        create_dataset(group, f'm{part_type}_{aperture_name}_gradient', shape=(group[f'm{part_type}_{aperture_name}'].shape[0],), dtype='f8', units='Msun/yr')
                        group[f'm{part_type}_{aperture_name}_gradient'].resize((group[f'm{part_type}_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        mass = unyt.unyt_array(values, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        time = unyt.unyt_array(group['age'], group['age'].attrs['units'])
                        gradient = np.gradient(mass.value, time.value, edge_order=1)
                        gradient = unyt.unyt_array(gradient, mass.units/time.units)
                        group[f'm{part_type}_{aperture_name}_gradient'][:] = gradient.in_units('Msun/yr')
                    except Exception as error:
                        print(f'\nError calculating {central_type} m{part_type}_{aperture_name}_gradient: {error}\n')

                    try:
                        create_dataset(group, f'm{part_type}_{aperture_name}_gradient_over_mstar', shape=(group[f'm{part_type}_{aperture_name}_gradient'].shape[0],), dtype='f8', units='yr**-1')
                        group[f'm{part_type}_{aperture_name}_gradient_over_mstar'].resize((group[f'm{part_type}_{aperture_name}_gradient'].shape[0],))
                        sfr = np.nan_to_num(group[f'm{part_type}_{aperture_name}_gradient'][:], nan=0.0, copy=True)
                        sfr = unyt.unyt_array(sfr, group[f'm{part_type}_{aperture_name}_gradient'].attrs['units'])
                        stellar_mass = np.nan_to_num(group[f'm{part_type}_{aperture_name}'][:], nan=0.0, copy=True)
                        stellar_mass = unyt.unyt_array(stellar_mass, group[f'm{part_type}_{aperture_name}'].attrs['units'])
                        ssfr = sfr / stellar_mass
                        group[f'm{part_type}_{aperture_name}_gradient_over_mstar'][:] = ssfr.in_units('yr**-1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} m{part_type}_{aperture_name}_gradient_over_mstar: {error}\n')


                if part_type == 'PartType5':
                    try:
                        create_dataset(group, f'f_{part_type}_phys_{aperture_name}-total', shape=(group[f'm{part_type}_phys_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_{part_type}_phys_{aperture_name}-total'].resize((group[f'm{part_type}_phys_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_phys_{aperture_name}'][:], nan=0.0, copy=True)
                        mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_phys_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'm_total_{aperture_name}'], group[f'm_total_{aperture_name}'].attrs['units'])
                        group[f'f_{part_type}_phys_{aperture_name}-total'][:] = mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} f_{part_type}_phys_{aperture_name}-total: {error}\n')

                    try:
                        create_dataset(group, f'f_{part_type}_phys_{aperture_name}-gas', shape=(group[f'm{part_type}_phys_{aperture_name}'].shape[0],), dtype='f8', units='1')
                        group[f'f_{part_type}_phys_{aperture_name}-gas'].resize((group[f'm{part_type}_phys_{aperture_name}'].shape[0],))
                        values = np.nan_to_num(group[f'm{part_type}_phys_{aperture_name}'][:], nan=0.0, copy=True)
                        mass_fraction = unyt.unyt_array(values, group[f'm{part_type}_phys_{aperture_name}'].attrs['units'])/unyt.unyt_array(group[f'mPartType0_{aperture_name}'], group[f'mPartType0_{aperture_name}'].attrs['units'])
                        group[f'f_{part_type}_phys_{aperture_name}-gas'][:] = mass_fraction.in_units('1')
                    except Exception as error:
                        print(f'\nError calculating {central_type} f_{part_type}_phys_{aperture_name}-gas: {error}\n')


print('\n\nDone!\n')