## module load NiaEnv/2022a python/3.11.5
## Python evironment: gen_profiles

import sys
import os
import argparse
import h5py
import gc
import copy
# from timeit import default_timer as timer

import yt
import unyt
import caesar
# from caesar.hydrogen_mass_calc import get_aperture_masses
import numpy as np

import pprint

gc.isenabled()



parser = argparse.ArgumentParser(prog='track_halo_properties.py', description='Track properties of halo and central galaxy across snapshots using supplied progenitors/descendants.')
parser.add_argument('--snap_dir', action='store', type=str, required=True, 
                    help='directory containing snapshots')
parser.add_argument('--snap_base', action='store', type=str, default='snapshot_',
                    help='base name for snapshots, e.g. snapshot_')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory containing caesar files')
parser.add_argument('--caesar_base', action='store', type=str, default='caesar_',
                    help='base name for caesar files, e.g. caesar_')
parser.add_argument('--caesar_suffix', action='store', type=str, default='',
                    help='suffix for caesar files, e.g. _haloid-fof_lowres-[2]')
# parser.add_argument('--source_snap_num', action='store', type=int, required=True, 
#                     help='Snapshot number for halo of which to find progenitor/descendant properties')
parser.add_argument('--target_snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers in which to find halo progenitors/descendants')
# parser.add_argument('--source_halo_id', action='store', type=int, required=True, 
#                     help='Id of source halo')
parser.add_argument('--target_halo_ids', action='store', nargs='*', type=int, required=True, 
                    help='Ids of target halos')
# parser.add_argument('--n_most', action='store', type=int, default=1, choices=[None, 1, 2],
#                     help='caesar progen n_most option; find n_most progenitors/descendents (None = all)')
parser.add_argument('--sim_model', action='store', type=str, choices=['Simba', 'Simba-C', 'Obsidian'], required=True,
                    help='Galaxy formation model of the simulation (for black hole feedback criteria)')

# parser.add_argument('--track_halos', action=argparse.BooleanOptionalAction, default=True, 
#                     help='Track properties of halos and their central galaxies')
# parser.add_argument('--track_centrals', action=argparse.BooleanOptionalAction, default=True, 
#                     help='Track properties of central galaxies and their parent halos')

parser.add_argument('--nproc', action='store', type=int, default=1,
                    help='caesar progen nproc option')

parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
                    help='Whether to clear the output file initially before writing to it')
args = parser.parse_args()

# print(args.n_most)


if not os.path.exists(args.output_file):
    print('Making output path and file')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    f = h5py.File(args.output_file, 'a')
    f.close()
    print()

if args.clear_output_file:
    print('Clearing output file')
    # save_object_with_dill(init_dict, args.output_file, mode='wb')
    f = h5py.File(args.output_file, 'w')
    f.close()
    # f = open(args.output_file, 'w')
    # f.close()
    print()







# def save_object_with_h5py(obj, filename, mode='a'):
#     with h5py.File(filename, mode) as f:  # mode='wb' overwrites any existing file.
#         dill.dump(obj, f, dill.HIGHEST_PROTOCOL)

def euclidean_distance(a, b):
    assert np.shape(a) == np.shape(b), f'Shapes of a and b are different'
    return(np.sqrt(np.sum((a-b)**2, axis=np.ndim(a)-1, keepdims=True)))

def create_group(file, group):
    if group not in file:
        file.create_group(group)

def create_dataset(group, dataset, shape=(0,), maxshape=(None,), dtype='f8', units=None):
    # print('start')
    if dataset not in group:
        print(group, dataset)
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

## Sphere for checking mass ratios of halos with target halo
sphere_radius_type = 'r500c'
sphere_radius_units = 'kpc'
sphere_radius_factor = 10.
mass_ratio_type = 'm500c'
major_merger_mass_ratio = 5.


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




part_types = ['PartType0', 'PartType1', 'PartType2', 'PartType4', 'PartType5']
# apertures = ['30 kpccm', '50 kpccm']
# aperture_names = ['30ckpc', '50ckpc']


delta_values = ['2500', '500', '200']
virial_quantities = ['circular_velocity', 'spin_param', 'temperature']
halo_sfr_types = ['', '_100']
# halo_mass_types = ['gas', 'stellar', 'baryon', 'dm', 'dm2', 'dust', 'bh', 'H2', 'H2_ism', 'HI', 'HI_ism']
halo_mass_types = ['H2', 'H2_ism', 'HI', 'HI_ism', 'baryon', 'bh', 'dm', 'dm2', 'dust', 'gas', 'stellar', 'total']
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
central_radii_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_radii_XX = ['half_mass', 'r20', 'r80']
central_sfr_types = ['', '_100']
central_metallicity_types = ['mass_weighted', 'sfr_weighted', 'stellar']
central_velocity_dispersion_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_age_types = ['mass_weighted', 'metal_weighted']
central_temperature_types = ['mass_weighted', 'mass_weighted_cgm']#, 'temp_weighted_cgm']
# central_rotation_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# central_rotation_XX = ['L', 'ALPHA', 'BETA', 'BoverT', 'kappa_rot']



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
if args.sim_model.lower() == 'manhattan':
    # Simba jet feedback criteria
    bh_mass_jet_min = unyt.unyt_quantity(10**7.5, 'Msun')
    bh_fedd_jet_max = unyt.unyt_quantity(0.2, '1')


###########################################################################################



# ## Load source snapshot
# source_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{args.source_snap_num:03}.hdf5')
# # try:
# source_snap = yt.load(source_snap_file)
# z_source = source_snap.current_redshift
# # except Exception as error:
# #     z_source = -1
# #     print(f'Error occurred loading source snapshot: {error}')

# ## Load source caesar file
# source_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{args.source_snap_num:03}{args.caesar_suffix}.hdf5')
# # try:
# source_obj = caesar.load(source_caesar_file)
# # except Exception as error:
# #     print(f'Error occurred loading source caesar file: {error}')
# #     print()
# #     continue
# print(f'Source caesar file: {source_caesar_file}, z={z_source}\n')

# print(f'Source halo id: {args.source_halo_id}')
# source_halo = source_obj.halos[args.source_halo_id]

# try:
#     source_central = source_halo.central_galaxy
#     source_central_id = source_central.GroupID
# except:
#     source_central = None
#     source_central_id = -1
# print(f'Source halo central galaxy id: {source_central_id}\n\n')


# with h5py.File(args.output_file, 'r+') as f:
    
## Loop through target caesar files
for target_snap_num, target_halo_id in zip(args.target_snap_nums, args.target_halo_ids):
    with h5py.File(args.output_file, 'r+') as f:
        print(f'\n\nTarget snap num: {target_snap_num}')
        print(f'\n\nTarget halo id: {target_halo_id}')
        print(type(target_snap_num))
        
        ## Load target snapshot
        target_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{target_snap_num:03}.hdf5')
        try:
            target_snap = yt.load(target_snap_file)
            units = target_snap.units
            # z_target = target_snap.current_redshift
            # target_data = target_snap.all_data()
            # apertures = [target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm')]
        except Exception as error:
            # z_target = -1
            print(f'Error occurred loading target snapshot: {error}')
            print()
            continue
            
        ## Load target caesar file
        target_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{target_snap_num:03}{args.caesar_suffix}.hdf5')
        try:
            target_obj = caesar.load(target_caesar_file)
            z_target = target_obj.simulation.redshift
            a_target = 1./(1. + z_target)
        except Exception as error:
            z_target = -1
            print(f'Error occurred loading target caesar file: {error}')
            print()
            continue
        print(f'Target caesar file: {target_caesar_file}, z={z_target}\n')



        H0 = target_snap.hubble_constant * unyt.unyt_quantity(100, 'km/s/Mpc')  # Convert little-h to H0
        print(f'\nH0 = {H0}, {H0.units}\n')




        ## Add in aperture mass quantities for galaxies
        ## These had to be requested specifically when making the caesar files, which was not done
        # print('\nCalculating aperture masses...')
        # aperture_mass_types = ['gas', 'star', 'dm', 'bh', 'sfr', 'HI', 'H2']
        # galaxy_stellar_half_mass_radii = np.array([gal.radii['stellar_half_mass'].in_units('kpccm') for gal in target_obj.galaxies])
        # galaxy_stellar_r80_radii = np.array([gal.radii['stellar_r80'].in_units('kpccm') for gal in target_obj.galaxies])
        # masses_30ckpc = get_aperture_masses(target_snap_file, target_obj.galaxies, target_obj.halos,
        #                                     quantities=aperture_mass_types, aperture=30,
        #                                     projection=None, nproc=args.nproc)
        # masses_50ckpc = get_aperture_masses(target_snap_file, target_obj.galaxies, target_obj.halos,
        #                                     quantities=aperture_mass_types, aperture=50,
        #                                     projection=None, exclude=None, nproc=args.nproc)
        # print('done\n')



        ## Get target halo
        try:
            halo = target_obj.halos[target_halo_id]
            create_group(f, 'halo')
        except:
            halo = None


        ## Get central galaxy of target halo
        try:
            central = halo.central_galaxy
            central_id = central.GroupID
            create_group(f, 'central')
        except:
            central = None
            central_id = -1
        print(f'Halo central galaxy id: {central_id}')


        ## Get satellite galaxies of target halo
        if False:
            try:
                satellites = halo.satellite_galaxies
                satellite_ids = [sat.GroupID for sat in satellites]
                create_group(f, 'satellites')
            except:
                satellites = []
                satellite_ids = []
            print(f'\nHalo satellite galaxy ids: {satellite_ids}\n')






        # # Get cosmology (H0 in 1/s and OmegaM dimensionless) with fallbacks from yt dataset or caesar simulation
        # H0 = None
        # OmegaM = None

        # # Helper to coerce numeric/quantity to unyt quantity in km/s/Mpc then to 1/s
        # def _coerce_H0_to_1_per_s(val):
        #     try:
        #         # If it's an astropy or unyt quantity, try to use its value and units
        #         q = unyt.unyt_quantity(val)
        #         # If units are dimensionless and value ~ 0.x treat as little-h and convert
        #         if q.units == unyt.dimensionless:
        #             v = float(q)
        #             if 0.0 < v < 2.0:
        #                 q = unyt.unyt_quantity(v * 100.0, 'km/s/Mpc')
        #             else:
        #                 q = unyt.unyt_quantity(v, 'km/s/Mpc')
        #     except Exception:
        #         # val might be a plain number
        #         v = float(val)
        #         if 0.0 < v < 2.0:
        #             q = unyt.unyt_quantity(v * 100.0, 'km/s/Mpc')
        #         else:
        #             q = unyt.unyt_quantity(v, 'km/s/Mpc')
        #     return q.in_units('1/s')

        # # Try yt snapshot first
        # try:
        #     # yt dataset cosmology object (many frontends expose .cosmology)
        #     cosmo = getattr(target_snap, 'cosmology', None)
        #     params = getattr(target_snap, 'parameters', {}) or {}
        #     if cosmo is not None:
        #         # common attr names
        #         if hasattr(cosmo, 'hubble_constant'):
        #             H0 = getattr(cosmo, 'hubble_constant')
        #         elif hasattr(cosmo, 'H0'):
        #             H0 = getattr(cosmo, 'H0')
        #         elif hasattr(cosmo, 'h'):
        #             H0 = getattr(cosmo, 'h')
        #         # OmegaM variants
        #         if hasattr(cosmo, 'Om0'):
        #             OmegaM = getattr(cosmo, 'Om0')
        #         elif hasattr(cosmo, 'omega_matter'):
        #             OmegaM = getattr(cosmo, 'omega_matter')
        #     # dataset parameters fallback (strings/keys vary between frontends)
        #     if H0 is None:
        #         for key in ('H0', 'h', 'hubble_constant'):
        #             if key in params:
        #                 H0 = params[key]
        #                 break
        #     if OmegaM is None:
        #         for key in ('omega_matter', 'Omega_m', 'Omega_M', 'omega_M'):
        #             if key in params:
        #                 OmegaM = params[key]
        #                 break
        # except Exception:
        #     H0 = None
        #     OmegaM = None

        # # Try caesar simulation object if needed
        # if (H0 is None) or (OmegaM is None):
        #     sim = getattr(target_obj, 'simulation', None)
        #     if sim is not None:
        #         # try sim.cosmology first
        #         simcos = getattr(sim, 'cosmology', None)
        #         if simcos is not None:
        #             if H0 is None:
        #                 for attr in ('hubble_constant', 'H0', 'h'):
        #                     if hasattr(simcos, attr):
        #                         H0 = getattr(simcos, attr)
        #                         break
        #             if OmegaM is None:
        #                 for attr in ('Om0', 'omega_matter', 'Omega_m'):
        #                     if hasattr(simcos, attr):
        #                         OmegaM = getattr(simcos, attr)
        #                         break
        #         # direct sim attributes
        #         if H0 is None:
        #             for attr in ('hubble_constant', 'H0', 'h'):
        #                 if hasattr(sim, attr):
        #                     H0 = getattr(sim, attr)
        #                     break
        #         if OmegaM is None:
        #             for attr in ('Om0', 'omega_matter', 'Omega_m'):
        #                 if hasattr(sim, attr):
        #                     OmegaM = getattr(sim, attr)
        #                     break
        #         # simulation parameters dict (if present)
        #         simparams = getattr(sim, 'parameters', {}) or {}
        #         if H0 is None:
        #             for key in ('H0', 'h', 'hubble_constant'):
        #                 if key in simparams:
        #                     H0 = simparams[key]; break
        #         if OmegaM is None:
        #             for key in ('omega_matter', 'Omega_m', 'Omega_M'):
        #                 if key in simparams:
        #                     OmegaM = simparams[key]; break

        # # Final coercion / sensible defaults
        # try:
        #     if H0 is None:
        #         # default to 70 km/s/Mpc if nothing found
        #         H0 = unyt.unyt_quantity(68.0, 'km/s/Mpc').in_units('1/s')
        #     else:
        #         H0 = _coerce_H0_to_1_per_s(H0)
        # except Exception:
        #     H0 = unyt.unyt_quantity(68.0, 'km/s/Mpc').in_units('1/s')

        # try:
        #     if OmegaM is None:
        #         # default to 0.3
        #         OmegaM = 0.3
        #     else:
        #         # try to coerce to plain float (astropy/unyt should work via float())
        #         OmegaM = float(unyt.unyt_quantity(OmegaM))
        # except Exception:
        #     OmegaM = 0.3



        
    
        
        # ## Link halos in snapshots with caesar progen
        # # caesar.progen.check_if_progen_is_present(target_caesar_file, 'progen_halo_dm')
        # halo_progens = caesar.progen.progen_finder(obj_current=source_obj, obj_target=target_obj, 
        #                                       caesar_file=source_caesar_file, snap_dir=args.snap_dir,
        #                                       data_type='halo', part_type='dm', recompute=True,
        #                                       save=False, n_most=2, min_in_common=0.1, nproc=args.nproc,
        #                                       match_frac=True, reverse_match=False)
        # print()
        # print('\nHalo progens:')
        # print(halo_progens)
        # print()
        
        # ## For n_most=1
        # try:
        #     first_mmp_halo_id = halo_progens[0][args.source_halo_id][0]  # with match_frac=True
        # except:
        #     first_mmp_halo_id = -1

        # try:
        #     second_mmp_halo_id = halo_progens[0][args.source_halo_id][1]  # with match_frac=True
        # except:
        #     second_mmp_halo_id = -1
        # # target_halo_id = progens[source_halo_id][0]  # with match_frac=False
        
        # print(f'First MMP halo id: {first_mmp_halo_id}')
        # if first_mmp_halo_id < 0:
        #     first_mmp_halo = None
        #     print()
        # else:
        #     first_mmp_halo = target_obj.halos[first_mmp_halo_id]
        #     print(f"First MMP halo m500c: {first_mmp_halo.virial_quantities['m500c']}")
        #     try:
        #         print(f'First MMP halo contamination: {first_mmp_halo.contamination}\n')
        #     except:
        #         print(f'First MMP halo contamination: Not available\n')
        #     create_group(f, 'first_mmp_halo')
        #     # first_time(prop_dict, 'halo', init={})

        # print(f'Second MMP halo id: {second_mmp_halo_id}')
        # if second_mmp_halo_id < 0:
        #     second_mmp_halo = None
        #     print()
        # else:
        #     second_mmp_halo = target_obj.halos[second_mmp_halo_id]
        #     print(f"Second MMP halo m500c: {second_mmp_halo.virial_quantities['m500c']}")
        #     try:
        #         print(f'Second MMP halo contamination: {second_mmp_halo.contamination}\n')
        #     except:
        #         print(f'Second MMP halo contamination: Not available\n')
        #     create_group(f, 'second_mmp_halo')
    
        # try:
        #     first_mmp_halo_central = first_mmp_halo.central_galaxy
        #     first_mmp_halo_central_id = first_mmp_halo_central.GroupID
        #     create_group(f, 'first_mmp_halo_central')
        #     # first_time(prop_dict, 'halo_central', init={})
        # except:
        #     first_mmp_halo_central = None
        #     first_mmp_halo_central_id = -1
        # # if target_central is not None:
        # #     target_central_index = target_central.GroupID
        # print(f'First MMP halo central galaxy id: {first_mmp_halo_central_id}')

        # try:
        #     second_mmp_halo_central = second_mmp_halo.central_galaxy
        #     second_mmp_halo_central_id = second_mmp_halo_central.GroupID
        #     create_group(f, 'second_mmp_halo_central')
        # except:
        #     second_mmp_halo_central = None
        #     second_mmp_halo_central_id = -1
        # print(f'Second MMP halo central galaxy id: {second_mmp_halo_central_id}\n')
        
        # ## For any n_most
        # # target_halo_ids = progens[0][source_halo_id]#[0]  # with match_frac=True
        # # # target_halo_ids = progens[source_halo_id]#[0]  # with match_frac=False
        # # target_halos = [target_obj.halos[target_halo_id] for target_halo_id in target_halo_ids]
        # # target_halo_m500c = [target_halo.virial_quantities['m500c'] for target_halo in target_halos]
        # # target_halo_contamination = [target_halo.contamination for target_halo in target_halos]
        # # print(f'Target halo ids: {target_halo_ids}')
        # # print(f'Target halo m500c: {target_halo_m500c}')
        # # print(f'Target halo contamination: {target_halo_contamination}')
        # # print()


        # # print('\nHELLO 1\n')
    
    
        # ## Link galaxies in snapshots with caesar progen
        # # caesar.progen.check_if_progen_is_present(target_caesar_file, 'progen_halo_dm')
        # if source_central is not None:
        #     # print('\nHELLO 2\n')
        #     try:
        #         # print('\nHELLO 3\n')
        #         gal_progens = caesar.progen.progen_finder(obj_current=source_obj, obj_target=target_obj, 
        #                                               caesar_file=source_caesar_file, snap_dir=args.snap_dir,
        #                                               data_type='galaxy', part_type='star', recompute=True,
        #                                               save=False, n_most=2, min_in_common=0.1, nproc=args.nproc,
        #                                               match_frac=True, reverse_match=False)
        #         print()
        #         # print('\nHELLO 4\n')
        #     except Exception as error:
        #         print(f'\nError doing gal_progens: {error}\n')
        #         gal_progens = None
                
        #     print('\nGalaxy progens:')
        #     print(gal_progens)
        #     print()
            
        #     ## For n_most=1
        #     try:
        #         first_mmp_central_id = gal_progens[0][source_central_id][0]  # with match_frac=True
        #     except:
        #         first_mmp_central_id = -1
                
        #     try:
        #         second_mmp_central_id = gal_progens[0][source_central_id][1]  # with match_frac=True
        #     except:
        #         second_mmp_central_id = -1
        #     # target_halo_id = progens[source_halo_id][0]  # with match_frac=False
            
        #     if first_mmp_central_id < 0:
        #         first_mmp_central = None
        #     else:
        #         first_mmp_central = target_obj.galaxies[first_mmp_central_id]
        #         # print(f'Target central id: {target_central_id}\n')
        #         create_group(f, 'first_mmp_central')
        #         # first_time(prop_dict, 'central', init={})

        #     if second_mmp_central_id < 0:
        #         second_mmp_central = None
        #     else:
        #         second_mmp_central = target_obj.galaxies[second_mmp_central_id]
        #         create_group(f, 'second_mmp_central')
        
        #     try:
        #         first_mmp_central_halo = first_mmp_central.halo
        #         first_mmp_central_halo_id = first_mmp_central_halo.GroupID
        #         create_group(f, 'first_mmp_central_halo')
        #         # first_time(prop_dict, 'central_halo', init={})
        #     except Exception as error:
        #         print(f'\nError getting first_mmp_central_halo: {error}\n')
        #         first_mmp_central_halo = None
        #         first_mmp_central_halo_id = -1
        #     # print(f'Target central galaxy halo id: {target_central_halo_id}')
        #     # if target_central_halo is not None:
        #     #     print(f"Target central galaxy halo m500c: {target_central_halo.virial_quantities['m500c']}")
        #     #     print(f'Target central galaxy halo contamination: {target_central_halo.contamination}\n')

        #     try:
        #         second_mmp_central_halo = second_mmp_central.halo
        #         second_mmp_central_halo_id = second_mmp_central_halo.GroupID
        #         create_group(f, 'second_mmp_central_halo')
        #     except Exception as error:
        #         print(f'\nError getting second_mmp_central_halo: {error}\n')
        #         second_mmp_central_halo = None
        #         second_mmp_central_halo_id = -1
                
        # else:
        #     gal_progens = None
            
        #     first_mmp_central_id = -1
        #     first_mmp_central = None

        #     second_mmp_central_id = -1
        #     second_mmp_central = None
            
        #     first_mmp_central_halo = None
        #     first_mmp_central_halo_id = -1

        #     second_mmp_central_halo = None
        #     second_mmp_central_halo_id = -1
    
        # print(f'First MMP central id: {first_mmp_central_id}')
        # print(f'Second MMP central id: {second_mmp_central_id}')
        # print(f'First MMP central galaxy halo id: {first_mmp_central_halo_id}')
        # print(f'Second MMP central galaxy halo id: {second_mmp_central_halo_id}\n')
        # if first_mmp_central_halo is not None:
        #     print(f"First MMP central galaxy halo m500c: {first_mmp_central_halo.virial_quantities['m500c']}")
        #     try:
        #         print(f'First MMP central galaxy halo contamination: {first_mmp_central_halo.contamination}\n')
        #     except:
        #         print(f'First MMP central galaxy halo contamination: Not available\n')
        # if second_mmp_central_halo is not None:
        #     print(f"Second MMP central galaxy halo m500c: {second_mmp_central_halo.virial_quantities['m500c']}")
        #     try:
        #         print(f'Second MMP central galaxy halo contamination: {second_mmp_central_halo.contamination}\n')
        #     except:
        #         print(f'Second MMP central galaxy halo contamination: Not available\n')
    
    
    
        # print()
        # pprint.pprint(prop_dict)
        # print()
    
    
        ## Get properties and save them to hdf5 file ########
        print('\nCalculating Halo Properties\n')
        # halos = [first_mmp_halo, second_mmp_halo, first_mmp_central_halo, second_mmp_central_halo]
        # centrals = [first_mmp_central, second_mmp_central, first_mmp_halo_central, second_mmp_halo_central]
        
        ## Halo properties
        # for halo_type, halo in zip(halo_types, halos):
        #     print(f'{halo_type}: {halo}\n')
            # if halo is None:
            #     continue
        if halo is not None:
    
            halo_type = 'halo'
            # print(f'\n{halo_type}\n')

            group = f[f'/{halo_type}']
            # group = f['/halo']
            # append_func = append_to_dataset
    
            # print()
            # pprint.pprint(prop_dict)
            # print()
            create_dataset(group, 'snap_num', dtype='i8', units='1')
            # first_time(prop_dict[halo_type], 'snap_num')
            curr_length = copy.deepcopy(len(group['snap_num'][:]))
            print()
            print(f"{halo_type} snap_num = {group['snap_num'][:]}")
            print(f'og curr_length = {curr_length}')
            print()
            if curr_length > 0:
                if target_snap_num in group['snap_num'][:-1]:
                    print(f'\nsnap_num {target_snap_num} for {halo_type} already saved - skipping\n')
                    continue
                elif target_snap_num == group['snap_num'][:][-1]:
                    print(f'\nsnap_num {target_snap_num} for {halo_type} likely only partly saved - overwriting properties\n')
                    curr_length -= 1
                    # append_func = replace_in_dataset
            print(f'curr_length = {curr_length}')
            # print()
            # pprint.pprint(prop_dict)
            # print()
            # prop_dict[halo_type]['snap_num'].append(unyt.unyt_array(target_snap_num, ''))
            print(f'target_snap_num: {target_snap_num}')
            append_to_dataset(group, 'snap_num', target_snap_num, curr_length)
            # print()
            # pprint.pprint(prop_dict)
            # print()
    
            # first_time(prop_dict[halo_type], 'age')
            create_dataset(group, 'age', dtype='f8', units='Gyr')
            # print()
            # pprint.pprint(prop_dict)
            # print()
            # prop_dict[halo_type]['age'].append(target_snap.current_time.in_units('Gyr'))
            append_to_dataset(group, 'age', target_snap.current_time.in_units('Gyr'), curr_length)
            # print()
            # pprint.pprint(prop_dict)
            # print()
    
            # sys.exit()
    
            # first_time(prop_dict[halo_type], 'z')
            create_dataset(group, 'z', dtype='f8', units='1')
            # prop_dict[halo_type]['z'].append(unyt.unyt_array(target_snap.current_redshift, ''))
            append_to_dataset(group, 'z', z_target, curr_length)
    
            # first_time(prop_dict[halo_type], 'id')
            create_dataset(group, 'id', dtype='i8', units='1')
            # prop_dict[halo_type]['id'].append(unyt.unyt_array(halo.GroupID, ''))
            append_to_dataset(group, 'id', halo.GroupID, curr_length)
    
            try:
                # first_time(prop_dict[halo_type], 'contamination')
                create_dataset(group, 'contamination', dtype='f8', units='1')
                # prop_dict[halo_type]['contamination'].append(unyt.unyt_array(halo.contamination, ''))
                append_to_dataset(group, 'contamination', halo.contamination, curr_length)
            except:
                pass

            for ii, coord in zip([0,1,2], ['x', 'y', 'z']):
                for unit, suffix in zip(['', 'cm'], ['phys', 'cm']):
                    create_dataset(group, f'minpotpos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
                    append_to_dataset(group, f'minpotpos_{coord}_{suffix}',
                                      halo.minpotpos.in_units(f'Mpc{unit}')[ii], curr_length)

                    create_dataset(group, f'compos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
                    append_to_dataset(group, f'compos_{coord}_{suffix}',
                                      halo.pos.in_units(f'Mpc{unit}')[ii], curr_length)
    
            # # first_time(prop_dict[halo_type], 'minpotpos')
            # create_dataset(group, 'minpotpos', shape=(0,3,), maxshape=(None,3,), dtype='f8')
            # # prop_dict[halo_type]['minpotpos'].append(halo.minpotpos.in_units('Mpccm'))
            # append_to_dataset(group, 'minpotpos', halo.minpotpos.in_units('Mpccm'))

            
            create_dataset(group, 'ngas', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'ngas', halo.ngas, curr_length)
            except:
                print(f'Bad {halo_type} ngas')
                append_to_dataset(group, 'ngas', 0., curr_length)

            create_dataset(group, 'nstar', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'nstar', halo.nstar, curr_length)
            except:
                print(f'Bad {halo_type} nstar')
                append_to_dataset(group, 'nstar', 0., curr_length)

            create_dataset(group, 'nbh', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'nbh', halo.nbh, curr_length)
            except:
                print(f'Bad {halo_type} nbh')
                append_to_dataset(group, 'nbh', 0., curr_length)

            create_dataset(group, 'ndm', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'ndm', halo.ndm, curr_length)
            except:
                print(f'Bad {halo_type} ndm')
                append_to_dataset(group, 'ndm', 0., curr_length)


            

            ## Halo virial temperature
            create_dataset(group, 'Tvir_vandeVoort+2011', dtype='f8', units='K')
            try:
                mu = 0.59 # Mean molecular weight for fully ionized gas with primordial composition
                # Tvir = (mu*unyt.mp*halo.virial_quantities['vmax']**2/(2*unyt.kb)).in_units('K')
                # van de Voort et al. (2011) definition
                Tvir_vandeVoort2011 = (((unyt.G**2 * H0**2 * target_snap.omega_matter * 18 * np.pi**2)/54)**(1./3.) * (mu * unyt.mh / unyt.kb) * (halo.virial_quantities['m200c']**(2./3.)) * (1 + z_target)).in_units('K')
                append_to_dataset(group, 'Tvir_vandeVoort+2011', Tvir_vandeVoort2011, curr_length)
            except Exception as error:
                print(f'Bad {halo_type} Tvir_vandeVoort+2011')
                print(f'Error: {error}')
                append_to_dataset(group, 'Tvir_vandeVoort+2011', np.nan, curr_length)
                # del Tvir_vandeVoort2011  # to avoid confusion later
                Tvir_vandeVoort2011 = np.nan  # to avoid confusion later

            # create_dataset(group, 'Tvir_van_ve_Voort_2019_v2', dtype='f8', units='K')
            # try:
            #     mu = 0.59 # Mean molecular weight for fully ionized gas with primordial composition
            #     # van de Voort et al. (2011) simplified definition wih constants combined
            #     Tvir_van_ve_Voort_2019_v2 = unyt.unyt_quantity(3e5, 'K') * (mu/0.59) * (halo.virial_quantities['m200c']/unyt.unyt_quantity(1e12, 'Msun'))**(2./3.) * (1+z_target)
            #     append_to_dataset(group, 'Tvir_van_ve_Voort_2019_v2', Tvir_van_ve_Voort_2019_v2, curr_length)
            # except:
            #     print(f'Bad {halo_type} Tvir_van_ve_Voort_2019_v2')
            #     append_to_dataset(group, 'Tvir_van_ve_Voort_2019_v2', np.nan, curr_length)

            create_dataset(group, 'Tvir_dave+2019', dtype='f8', units='K')
            try:
                # Voit (2005) simplified definition from dave+2019
                Tvir_dave2019 = unyt.unyt_quantity(9.52e7, 'K') * (halo.virial_quantities['m200c']/unyt.unyt_quantity(1e15, 'Msun'))**(1./3.)
                append_to_dataset(group, 'Tvir_dave+2019', Tvir_dave2019, curr_length)
            except:
                print(f'Bad {halo_type} Tvir_dave+2019')
                append_to_dataset(group, 'Tvir_dave+2019', np.nan, curr_length)
                # del Tvir_dave2019  # to avoid confusion later
                Tvir_dave2019 = np.nan  # to avoid confusion later





            ## Gas phase definitions
            gas_phases = {
                'Sokolowska+2018':{
                    'cold':{
                        'temperature':{
                            '<=': unyt.unyt_quantity(3e4, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'warm':{
                        'temperature':{
                            '>=': unyt.unyt_quantity(3e4, 'K'),
                            '<=': unyt.unyt_quantity(1e5, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'warm_hot':{
                        'temperature':{
                            '>=': unyt.unyt_quantity(1e5, 'K'),
                            '<=': unyt.unyt_quantity(1e6, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'hot':{
                        'temperature':{
                            '>=': unyt.unyt_quantity(1e6, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                },
                'vandeVoort+2011':{
                    'diffuse_IGM':{
                        'density':{
                            '<=': 1e2 * target_snap.critical_density,
                        },
                        'temperature':{
                            '<=': unyt.unyt_quantity(1e5, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'cold_halo_gas':{
                        'density':{
                            '>=': 1e2 * target_snap.critical_density,
                        },
                        'H_nuclei_density':{
                            '<=': unyt.unyt_quantity(0.13, 'cm**-3'),
                        },
                        'temperature':{
                            '<=': unyt.unyt_quantity(1e5, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'WHIM':{
                        'H_nuclei_density':{
                            '<=': unyt.unyt_quantity(0.13, 'cm**-3'),
                        },
                        'temperature':{
                            '>=': unyt.unyt_quantity(1e5, 'K'),
                            '<=': unyt.unyt_quantity(1e7, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'ICM':{
                        'H_nuclei_density':{
                            '<=': unyt.unyt_quantity(0.13, 'cm**-3'),
                        },
                        'temperature':{
                            '>=': unyt.unyt_quantity(1e7, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'star_forming_ISM':{
                        'H_nuclei_density':{
                            '>=': unyt.unyt_quantity(0.13, 'cm**-3'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'sub_virial':{
                        'temperature':{
                            '<=': Tvir_vandeVoort2011,
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'super_virial':{
                        'temperature':{
                            '>=': Tvir_vandeVoort2011,
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                },
                'dave+2019':{
                    'sub_virial':{
                        'temperature':{
                            '<=': Tvir_dave2019,
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'super_virial':{
                        'temperature':{
                            '>=': Tvir_dave2019,
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                },
                'Aviv':{
                    'coupled_gas':{
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'decoupled_gas':{
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'launched_wind':{
                        'NWindLaunches':{
                            '>=': unyt.unyt_quantity(1, ''),
                        },
                    },
                    'unlaunched_wind':{
                        'NWindLaunches':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>0':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                    },
                    'outflow>0_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>0_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>0':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                    },
                    'inflow>0_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>0_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>1000':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1e3, 'km/s'),
                        },
                    },
                    'outflow>1000_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>1000_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>1000':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1e3, 'km/s'),
                        },
                    },
                    'inflow>1000_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>1000_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>5000':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(5e3, 'km/s'),
                        },
                    },
                    'outflow>5000_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(5000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>5000_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(5000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>5000':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-5e3, 'km/s'),
                        },
                    },
                    'inflow>5000_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-5000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>5000_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-5000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>10000':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1e4, 'km/s'),
                        },
                    },
                    'outflow>10000_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1e4, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow>10000_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1e4, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>10000':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1e4, 'km/s'),
                        },
                    },
                    'inflow>10000_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1e4, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow>10000_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1e4, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow0-300':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(300, 'km/s'),
                        },
                    },
                    'outflow0-300_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(300, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow0-300_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(300, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow300-1000':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(300, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(1000, 'km/s'),
                        },
                    },
                    'outflow300-1000_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(300, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow300-1000_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(300, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow1000-10000':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(10000, 'km/s'),
                        },
                    },
                    'outflow1000-10000_nowind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(10000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'outflow1000-10000_onlywind':{
                        'radial_velocity':{
                            '>': unyt.unyt_quantity(1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '<=': unyt.unyt_quantity(10000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },

                    'inflow0-300':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-300, 'km/s'),
                        },
                    },
                    'inflow0-300_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-300, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow0-300_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(0, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-300, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow300-1000':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-300, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                    },
                    'inflow300-1000_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-300, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow300-1000_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-300, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow1000-10000':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-10000, 'km/s'),
                        },
                    },
                    'inflow1000-10000_nowind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-10000, 'km/s'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                    'inflow1000-10000_onlywind':{
                        'radial_velocity':{
                            '<': unyt.unyt_quantity(-1000, 'km/s'),
                        },
                        'radial_velocity':{
                            '>=': unyt.unyt_quantity(-10000, 'km/s'),
                        },
                        'DelayTime':{
                            '>': unyt.unyt_quantity(0, ''),
                        },
                    },

                    'IGrM':{
                        'H_nuclei_density':{
                            '<=': unyt.unyt_quantity(0.13, 'cm**-3'),
                        },
                        'temperature':{
                            '>=': unyt.unyt_quantity(5e5, 'K'),
                        },
                        'DelayTime':{
                            '==': unyt.unyt_quantity(0, ''),
                        },
                    },
                },
            }

            # try:
            #     gas_phases['vandeVoort+2011']['sub_virial'] = {
            #         'temperature':{
            #             '<=': Tvir_vandeVoort2011,
            #         },
            #         'DelayTime':{
            #             '==': unyt.unyt_quantity(0, ''),
            #         },
            #     },
            #     gas_phases['vandeVoort+2011']['super_virial'] = {
            #         'temperature':{
            #             '>=': Tvir_vandeVoort2011,
            #         },
            #         'DelayTime':{
            #             '==': unyt.unyt_quantity(0, ''),
            #         },
            #     },
            # except:
            #     print('Cannot define sub-/super-virial gas phases due to missing Tvir_vandeVoort+2011')

            # try:
            #     gas_phases['Aviv'] = {
            #         'sub_virial_van_de_Voort_2019_v2':{
            #             'temperature':{
            #                 '<=': Tvir_van_ve_Voort_2019_v2,
            #             },
            #             'DelayTime':{
            #                 '==': unyt.unyt_quantity(0, ''),
            #             },
            #         },
            #         'super_virial_van_de_Voort_2019_v2':{
            #             'temperature':{
            #                 '>=': Tvir_van_ve_Voort_2019_v2,
            #             },
            #             'DelayTime':{
            #                 '==': unyt.unyt_quantity(0, ''),
            #             },
            #         },
            #     }
            # except:
            #     print('Cannot define sub-/super-virial gas phases due to missing Tvir_van_ve_Voort_2019_v2')

            # try:
            #     # gas_phases['dave+2019'] = {}
            #     gas_phases['dave+2019']['sub_virial'] = {
            #         'temperature':{
            #             '<=': Tvir_dave2019,
            #         },
            #         'DelayTime':{
            #             '==': unyt.unyt_quantity(0, ''),
            #         },
            #     },
            #     gas_phases['dave+2019']['super_virial'] = {
            #         'temperature':{
            #             '>=': Tvir_dave2019,
            #         },
            #         'DelayTime':{
            #             '==': unyt.unyt_quantity(0, ''),
            #         },
            #     },
            # except:
            #     print('Cannot define sub-/super-virial gas phases due to missing Tvir_dave+2019')

            



            ## Get centre of halo (minpotpos) and bulk velocity of halo (comvel) from caesar
            ## for getting velocities relative to halo bulk velocity
            minpotpos = halo.minpotpos
            comvel = halo.vel


            ## Calculate properties within different apertures
            aperture_names = ['3ckpc', '30ckpc', '50ckpc'] + [f'r{delta_value}c' for delta_value in delta_values] + ['0.5r200c', '0.1r200c']
            apertures = [target_snap.arr(3, 'kpccm'), target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm')] + [halo.virial_quantities[f'r{delta_value}c'].in_units('kpccm') for delta_value in delta_values] + [0.5*halo.virial_quantities['r200c'].in_units('kpccm'), 0.1*halo.virial_quantities['r200c'].in_units('kpccm')]
            for aperture, aperture_name in zip(apertures, aperture_names):
                print(f'\n\nCalculating properties within aperture: {aperture_name} = {aperture}\n')
                sphere = target_snap.sphere(minpotpos, aperture)
                sphere.set_field_parameter('center', minpotpos)
                sphere.set_field_parameter('bulk_velocity', comvel)

                for part_type in part_types:

                    create_dataset(group, f'n{part_type}_{aperture_name}', dtype='f8', units='1')
                    try:
                        npart = len(sphere[part_type, 'Masses'])
                        append_to_dataset(group, f'n{part_type}_{aperture_name}', npart, curr_length)
                    except:
                        print(f'Bad {halo_type} n{part_type}_{aperture_name}')
                        append_to_dataset(group, f'n{part_type}_{aperture_name}', 0., curr_length)

                    create_dataset(group, f'm{part_type}_{aperture_name}', dtype='f8', units='Msun')
                    try:
                        mpart = sum(sphere[part_type, 'Masses'].in_units('Msun'))
                        append_to_dataset(group, f'm{part_type}_{aperture_name}', mpart, curr_length)
                    except:
                        print(f'Bad {halo_type} m{part_type}_{aperture_name}')
                        append_to_dataset(group, f'm{part_type}_{aperture_name}', 0., curr_length)



                    if part_type == 'PartType0':
                        ## Cumulative number of wind launches
                        create_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', dtype='f8', units='1')
                        try:
                            nwind_launches = sum(sphere[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {halo_type} {part_type}_NWindLaunches_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', dtype='f8', units='1')
                        try:
                            sphere_nowinds = sphere.cut_region([f"obj['{part_type}', 'DelayTime'].in_units('') == 0"])
                            nwind_launches = sum(sphere_nowinds[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {halo_type} {part_type}_NWindLaunches_nowind_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', dtype='f8', units='1')
                        try:
                            sphere_windsonly = sphere.cut_region([f"obj['{part_type}', 'DelayTime'].in_units('') > 0"])
                            nwind_launches = sum(sphere_windsonly[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {halo_type} {part_type}_NWindLaunches_onlywind_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', 0., curr_length)
                        


                        ## Mass of gas in different phases
                        for phase_def_name, phase_defs in gas_phases.items():
                            print(f'\nCalculating gas phases for definition: {phase_def_name}\n')
                            for phase_name, phase_criteria in phase_defs.items():
                                print(f'Phase: {phase_name}')
                                # phase_filter = np.ones(len(sphere[part_type, 'Masses']), dtype=bool)
                                phase_filter = []
                                print(phase_criteria)
                                for field_name, limits in phase_criteria.items():
                                    print(f'  Field: {field_name}')
                                    for limit_type, limit_value in limits.items():
                                        phase_filter.append(f"obj['{part_type}', '{field_name}'].in_units('{limit_value.units}') {limit_type} {limit_value.value}")

                                    # field_values = sphere[part_type, field_name]
                                    # if 'min' in limits:
                                    #     phase_filter &= (field_values >= limits['min'])
                                    # if 'max' in limits:
                                    #     phase_filter &= (field_values <= limits['max'])

                                dataset_name = f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'
                                create_dataset(group, dataset_name, dtype='f8', units='Msun')
                                try:
                                    sphere_phase = sphere.cut_region(phase_filter) ## !! does cut region inherit field parameters from parent sphere?? !! ##
                                    mpart_phase = sum(sphere_phase[part_type, 'Masses'].in_units('Msun'))
                                    append_to_dataset(group, dataset_name, mpart_phase, curr_length)
                                except:
                                    print(f'Bad {halo_type} {dataset_name}')
                                    append_to_dataset(group, dataset_name, 0., curr_length)

                                # try:
                                #     mpart_phase = sum(sphere[part_type, 'Masses'][phase_filter].in_units('Msun'))
                                #     append_to_dataset(group, dataset_name, mpart_phase, curr_length)
                                # except:
                                #     print(f'Bad {halo_type} {dataset_name}')
                                #     append_to_dataset(group, dataset_name, np.nan, curr_length)

                        
                        ## Instantaneous star formation rate
                        create_dataset(group, f'sfr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
                        try:
                            sfr = sum(sphere[part_type, 'StarFormationRate'].in_units('Msun/yr'))
                            append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', sfr, curr_length)
                        except:
                            print(f'Bad {halo_type} sfr_{part_type}_{aperture_name}')
                            append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', 0., curr_length)




                    if part_type == 'PartType4':
                        ## Time-averaged star formation rate
                        create_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
                        try:
                            # formation_times = sphere[part_type, 'StellarFormationTime']
                            ages = sphere[part_type, 'age']
                            # current_time = target_snap.current_time.in_units('Gyr')
                            # age_100Myr_filter = np.where((current_time - formation_times.in_units('Gyr')) <= 0.1)[0]
                            age_100Myr_filter = np.where(ages.in_units('Gyr') <= 0.1)[0]
                            masses = sphere[part_type, 'Masses'][age_100Myr_filter].in_units('Msun')
                            summed_mass = unyt.unyt_array(sum(masses), units='Msun')
                            time_diff = unyt.unyt_quantity(0.1, 'Gyr')
                            sfr_100Myr = summed_mass / time_diff.in_units('yr')
                            # sfr_100Myr = sum(sphere[part_type, 'Masses'][age_100Myr_filter].in_units('Msun')) / 0.1
                            append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', sfr_100Myr, curr_length)
                        except Exception as error:
                            print(f'Bad {halo_type} sfr_100Myr_{part_type}_{aperture_name}')
                            print(error)
                            append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', 0., curr_length)
                        


                    
                    if part_type == 'PartType5':

                        create_dataset(group, f'm{part_type}_phys_{aperture_name}', dtype='f8', units='Msun')
                        try:
                            mpart = sum(sphere[part_type, 'BH_Mass'].in_units('Msun'))
                            append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', mpart, curr_length)
                        except:
                            print(f'Bad {halo_type} m{part_type}_phys_{aperture_name}')
                            append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', 0., curr_length)


                        # create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')

                        # bh_properties_good = True
                        try:
                            ## Find Mbh,phys and f_edd for each BH
                            bh_mass = sphere[part_type, 'BH_Mass'].in_units('Msun')
                            bh_mdot = sphere[part_type, 'BH_Mdot'].in_units('Msun/yr')
                            # f_rad = 0.1
                            bh_mdot_acc = bh_mdot / (1 - f_rad) # Full accretion rate onto the BH not accounting for radiative losses
                            bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson)).in_units('Msun/yr')
                            bh_fedd = (bh_mdot/bh_mdot_edd).in_units('1')
                            print('\nbh_mass:')
                            print(bh_mass)
                            print()
                            print('\nbh_mdot:')
                            print(bh_mdot)
                            print()
                            print('\nbh_mdot_edd:')
                            print(bh_mdot_edd)
                            print()
                            print('\nbh_fedd:')
                            print(bh_fedd)
                            print()

                        except Exception as error:
                            print(f'Bad {halo_type} {aperture_name} BH properties calculation')
                            # bh_properties_good = False
                            print(error)
                            print()
                            try:
                                del bh_mass
                            except:
                                pass
                            try:
                                del bh_mdot
                            except:
                                pass
                            try:
                                del bh_mdot_acc
                            except:
                                pass
                            try:
                                del bh_mdot_edd
                            except:
                                pass
                            try:
                                del bh_fedd
                            except:
                                pass



                        create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_quasar_high_fedd_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', dtype='f8', units='1')
                        # if not bh_properties_good:
                        try:
                            
                            # Find BHs with:
                            # No accretion (f_edd = 0);
                            # AGN in radiative/quasar mode (f_edd > 0.2);
                            # AGN in jet mode (f_edd <= 0.2) & Mbh,phys >= Mbh,jet,min (4e7 Msun for Simba, 7e7 Msun for Simba-C)
                            # (see e.g., Dave et al. 2019, Angles-Alcazar et al. 2020, Thomas et al. 2021, and references therein)
                            # bh_mass_jet_min = target_snap.arr(7e7, 'Msun')
                            # bh_fedd_jet_max = target_snap.arr(0.2, '1')

                            bh_no_accretion_filter = np.where(bh_fedd == unyt.unyt_quantity(0, ''))[0]


                            bh_quasar_high_fedd_filt = bh_fedd >= bh_fedd_jet_max
                            bh_quasar_low_fedd_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
                            bh_quasar_low_fedd_ascale_filt = np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
                            bh_quasar_fedd002_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
                            bh_quasar_fedd002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
                            bh_quasar_fedd0002_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))
                            bh_quasar_fedd0002_ascale_filt = np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, ''))

                            bh_quasar_filter = np.where(np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_filt))[0]
                            bh_quasar_ascale_filter = np.where(np.logical_or(bh_quasar_high_fedd_filt, bh_quasar_low_fedd_ascale_filt))[0]
                            # bh_quasar_filter = np.where(np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= bh_mass_jet_min))[0]
                            # bh_quasar_ascale_filter = np.where(np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= a_target*bh_mass_jet_min))[0]

                            bh_quasar_high_fedd_filter = np.where(bh_quasar_high_fedd_filt)[0]

                            bh_quasar_low_fedd_filter = np.where(bh_quasar_low_fedd_filt)[0]
                            bh_quasar_low_fedd_ascale_filter = np.where(bh_quasar_low_fedd_ascale_filt)[0]

                            bh_quasar_fedd002_filter = np.where(bh_quasar_fedd002_filt)[0]
                            bh_quasar_fedd002_ascale_filter = np.where(bh_quasar_fedd002_ascale_filt)[0]
                            
                            bh_quasar_fedd0002_filter = np.where(bh_quasar_fedd0002_filt)[0]
                            bh_quasar_fedd0002_ascale_filter = np.where(bh_quasar_fedd0002_ascale_filt)[0]

                            bh_jet_filter = np.where(np.logical_and(np.logical_and(bh_mass > bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, '')))[0]
                            bh_jet_ascale_filter = np.where(np.logical_and(np.logical_and(bh_mass > a_target*bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, '')))[0]


                            print('\nbh_no_accretion_filter:')
                            print(bh_no_accretion_filter)

                            print('\nbh_quasar_filter:')
                            print(bh_quasar_filter)
                            print('\nbh_quasar_ascale_filter:')
                            print(bh_quasar_ascale_filter)

                            print('\nbh_quasar_high_fedd_filter:')
                            print(bh_quasar_high_fedd_filter)

                            print('\nbh_quasar_low_fedd_filter:')
                            print(bh_quasar_low_fedd_filter)
                            print('\nbh_quasar_low_fedd_ascale_filter:')
                            print(bh_quasar_low_fedd_ascale_filter)

                            print('\nbh_quasar_fedd<0.02_filter:')
                            print(bh_quasar_fedd002_filter)
                            print('\nbh_quasar_fedd<0.02_ascale_filter:')
                            print(bh_quasar_fedd002_ascale_filter)

                            print('\nbh_quasar_fedd<0.002_filter:')
                            print(bh_quasar_fedd0002_filter)
                            print('\nbh_quasar_fedd<0.002_ascale_filter:')
                            print(bh_quasar_fedd0002_ascale_filter)

                            print('\nbh_jet_filter:')
                            print(bh_jet_filter)
                            print('\nbh_jet_ascale_filter:')
                            print(bh_jet_ascale_filter)
                            print()


                            nbh_no_accretion = len(bh_no_accretion_filter)

                            nbh_quasar = len(bh_quasar_filter)
                            nbh_quasar_ascale = len(bh_quasar_ascale_filter)

                            nbh_quasar_high_fedd = len(bh_quasar_high_fedd_filter)

                            nbh_quasar_low_fedd = len(bh_quasar_low_fedd_filter)
                            nbh_quasar_low_fedd_ascale = len(bh_quasar_low_fedd_ascale_filter)

                            nbh_quasar_fedd002 = len(bh_quasar_fedd002_filter)
                            nbh_quasar_fedd002_ascale = len(bh_quasar_fedd002_ascale_filter)

                            nbh_quasar_fedd0002 = len(bh_quasar_fedd0002_filter)
                            nbh_quasar_fedd0002_ascale = len(bh_quasar_fedd0002_ascale_filter)

                            nbh_jet = len(bh_jet_filter)
                            nbh_jet_ascale = len(bh_jet_ascale_filter)


                            append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', nbh_no_accretion, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', nbh_quasar, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', nbh_quasar_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', nbh_quasar_high_fedd, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', nbh_quasar_low_fedd, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', nbh_quasar_low_fedd_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', nbh_quasar_fedd002, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', nbh_quasar_fedd002_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', nbh_quasar_fedd0002, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', nbh_quasar_fedd0002_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', nbh_jet, curr_length)
                            append_to_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', nbh_jet_ascale, curr_length)

                        except Exception as error:
                            print(f'Bad {halo_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
                            print(error)
                            append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', 0., curr_length)



                        ## Calculate simple bolomentric AGN luminosity and jet power sums
                        create_dataset(group, f'bh_Lbol_{aperture_name}', dtype='f8', units='erg/s')
                        # create_dataset(group, f'Pjet_jets_{aperture_name}', dtype='f8', units='erg/s')
                        try:
                            Lbol = sum((f_rad * unyt.c**2 * bh_mdot).in_units('erg/s'))
                            append_to_dataset(group, f'bh_Lbol_{aperture_name}', Lbol, curr_length)
                        except:
                            print(f'Bad {halo_type} bh_Lbol_{aperture_name}')
                            append_to_dataset(group, f'bh_Lbol_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'bh_Lbol_acc_{aperture_name}', dtype='f8', units='erg/s')
                        try:
                            Lbol = sum((f_rad * unyt.c**2 * bh_mdot_acc).in_units('erg/s'))
                            append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', Lbol, curr_length)
                        except:
                            print(f'Bad {halo_type} bh_Lbol_acc_{aperture_name}')
                            append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', 0., curr_length)
                        
                        # try:
                        #     Pjet_jets = sum((unyt.epsilon_kin * unyt.c**2 * sphere[part_type, 'BH_Mdot'][bh_jet_filter]).in_units('erg/s'))
                        #     append_to_dataset(group, f'Pjet_jets_{aperture_name}', Pjet_jets, curr_length)
                        # except:
                        #     print(f'Bad {halo_type} Pjet_jets_{aperture_name}')
                        #     append_to_dataset(group, f'Pjet_jets_{aperture_name}', np.nan, curr_length)




                        # except Exception as error:
                        #     print(f'Bad {halo_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
                        #     print(error)
                        #     print()
                        #     append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', np.nan, curr_length)
                        #     append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', np.nan, curr_length)
                        #     append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', np.nan, curr_length)
                            
                        
                        
                
    
            for delta_value in delta_values:
                # first_time(prop_dict[halo_type], f'm{delta_value}c')
                create_dataset(group, f'm{delta_value}c', dtype='f8', units='Msun')
                # prop_dict[halo_type][f'm{delta_value}c'].append(halo.virial_quantities[f'm{delta_value}c'])
                append_to_dataset(group, f'm{delta_value}c', halo.virial_quantities[f'm{delta_value}c'].in_units('Msun'), curr_length)
    
                # first_time(prop_dict[halo_type], f'r{delta_value}c')
                create_dataset(group, f'r{delta_value}c', dtype='f8', units='kpc')
                # prop_dict[halo_type][f'r{delta_value}c'].append(halo.virial_quantities[f'r{delta_value}c'])
                append_to_dataset(group, f'r{delta_value}c', halo.virial_quantities[f'r{delta_value}c'].in_units('kpc'), curr_length)

                create_dataset(group, f'r{delta_value}c_cm', dtype='f8', units='kpccm')
                # prop_dict[halo_type][f'r{delta_value}c'].append(halo.virial_quantities[f'r{delta_value}c'])
                append_to_dataset(group, f'r{delta_value}c_cm', halo.virial_quantities[f'r{delta_value}c'].in_units('kpccm'), curr_length)
    
            for quant in virial_quantities:
                # print(quant)
                # print(str(halo.virial_quantities[f'{quant}'].units))
                # first_time(prop_dict[halo_type], f'{quant}')
                create_dataset(group, f'{quant}', dtype='f8', units=str(halo.virial_quantities[f'{quant}'].units))
                # prop_dict[halo_type][f'{quant}'].append(halo.virial_quantities[f'{quant}'])
                append_to_dataset(group, f'{quant}', halo.virial_quantities[f'{quant}'], curr_length)

            
            create_dataset(group, 'bh_mdot', dtype='f8', units='Msun/yr')
            try:
                append_to_dataset(group, 'bh_mdot', central.bhmdot.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {halo_type} bh_mdot')
                append_to_dataset(group, 'bh_mdot', 0., curr_length)
    
            create_dataset(group, 'bh_fedd', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'bh_fedd', central.bh_fedd.in_units(''), curr_length)
            except:
                print(f'Bad {halo_type} bh_fedd')
                append_to_dataset(group, 'bh_fedd', 0., curr_length)

    
            # first_time(prop_dict[halo_type], 'sfr')
            create_dataset(group, 'sfr', dtype='f8', units='Msun/yr')
            try:
                # prop_dict[halo_type]['sfr'].append(halo.sfr.in_units('Msun/yr'))
                append_to_dataset(group, 'sfr', halo.sfr.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {halo_type} sfr')
                # prop_dict[halo_type]['sfr'].append(unyt.unyt_array(0, 'Msun/yr'))
                # append_to_dataset(group, 'sfr', unyt.unyt_array(0, 'Msun/yr'))
                append_to_dataset(group, 'sfr', 0, curr_length)
    
            # first_time(prop_dict[halo_type], 'sfr_100')
            create_dataset(group, 'sfr_100', dtype='f8', units='Msun/yr')
            try:
                # prop_dict[halo_type]['sfr_100'].append(halo.sfr_100.in_units('Msun/yr'))
                append_to_dataset(group, 'sfr_100', halo.sfr_100.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {halo_type} sfr_100')
                # prop_dict[halo_type]['sfr_100'].append(unyt.unyt_array(0, 'Msun/yr'))
                append_to_dataset(group, 'sfr_100', 0, curr_length)
    
            for mass_type in halo_mass_types:
                # first_time(prop_dict[halo_type], f'{mass_type}_mass')
                create_dataset(group, f'{mass_type}_mass', dtype='f8', units='Msun')
                try:
                    # prop_dict[halo_type][f'{mass_type}_mass'].append(halo.masses[mass_type].in_units('Msun'))
                    append_to_dataset(group, f'{mass_type}_mass', halo.masses[mass_type].in_units('Msun'), curr_length)
                except:
                    print(f'Bad {halo_type} {mass_type}_mass')
                    # prop_dict[halo_type][f'{mass_type}_mass'].append(unyt.unyt_array(0, 'Msun'))
                    append_to_dataset(group, f'{mass_type}_mass', 0, curr_length)
    
            for radii_type in halo_radii_types:
                for XX in halo_radii_XX:
                    # first_time(prop_dict[halo_type], f'{radii_type}_{XX}_radius')
                    create_dataset(group, f'{radii_type}_{XX}_radius', dtype='f8', units='kpc')
                    try:
                        # prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(halo.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius', halo.radii[f'{radii_type}_{XX}'].in_units('kpc'), curr_length)
                    except:
                        print(f'Bad {halo_type} {radii_type}_{XX}_radius')
                        # prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius', np.nan, curr_length)

                    create_dataset(group, f'{radii_type}_{XX}_radius_cm', dtype='f8', units='kpccm')
                    try:
                        # prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(halo.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', halo.radii[f'{radii_type}_{XX}'].in_units('kpccm'), curr_length)
                    except:
                        print(f'Bad {halo_type} {radii_type}_{XX}_radius_cm')
                        # prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', np.nan, curr_length)
    
            for metal_type in halo_metallicity_types:
                # first_time(prop_dict[halo_type], f'{metal_type}_metallicity')
                create_dataset(group, f'{metal_type}_metallicity', dtype='f8', units='1')
                try:
                    # prop_dict[halo_type][f'{metal_type}_metallicity'].append(halo.metallicities[metal_type])
                    append_to_dataset(group, f'{metal_type}_metallicity', halo.metallicities[metal_type], curr_length)
                except:
                    print(f'Bad {halo_type} {metal_type}_metallicity')
                    # prop_dict[halo_type][f'{metal_type}_metallicity'].append(unyt.unyt_array(np.nan, ''))
                    append_to_dataset(group, f'{metal_type}_metallicity', np.nan, curr_length)
    
            for vel_disp_type in halo_velocity_dispersion_types:
                # first_time(prop_dict[halo_type], f'{vel_disp_type}_velocity_dispersion')
                create_dataset(group, f'{vel_disp_type}_velocity_dispersion', dtype='f8', units='km/s')
                try:
                    # prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(halo.velocity_dispersions[vel_disp_type].in_units('km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', halo.velocity_dispersions[vel_disp_type].in_units('km/s'), curr_length)
                except:
                    print(f'Bad {halo_type} {vel_disp_type}_velocity_dispersion')
                    # prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', np.nan, curr_length)

                create_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', dtype='f8', units='kmcm/s')
                try:
                    # prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(halo.velocity_dispersions[vel_disp_type].in_units('km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', halo.velocity_dispersions[vel_disp_type].in_units('kmcm/s'), curr_length)
                except:
                    print(f'Bad {halo_type} {vel_disp_type}_velocity_dispersion_cm')
                    # prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', np.nan, curr_length)
    
            for age_type in halo_age_types:
                # first_time(prop_dict[halo_type], f'{age_type}_stellar_age')
                create_dataset(group, f'{age_type}_stellar_age', dtype='f8', units='Gyr')
                try:
                    # prop_dict[halo_type][f'{age_type}_stellar_age'].append(halo.ages[age_type].in_units('Gyr'))
                    append_to_dataset(group, f'{age_type}_stellar_age', halo.ages[age_type].in_units('Gyr'), curr_length)
                except:
                    print(f'Bad {halo_type} {age_type}_stellar_age')
                    # prop_dict[halo_type][f'{age_type}_stellar_age'].append(unyt.unyt_array(np.nan, 'Gyr'))
                    append_to_dataset(group, f'{age_type}_stellar_age', np.nan, curr_length)
    
            for temp_type in halo_temperature_types:
                # first_time(prop_dict[halo_type], f'{temp_type}_temperature')
                create_dataset(group, f'{temp_type}_temperature', dtype='f8', units='K')
                try:
                    # prop_dict[halo_type][f'{temp_type}_temperature'].append(halo.temperatures[temp_type].in_units('K'))
                    append_to_dataset(group, f'{temp_type}_temperature', halo.temperatures[temp_type].in_units('K'), curr_length)
                except:
                    print(f'Bad {halo_type} {temp_type}_temperature')
                    # prop_dict[halo_type][f'{temp_type}_temperature'].append(unyt.unyt_array(np.nan, 'K'))
                    append_to_dataset(group, f'{temp_type}_temperature', np.nan, curr_length)
    
            for dens_type in halo_local_density_types:
                # first_time(prop_dict[halo_type], f'local_mass_density_{dens_type}kpccm')
                create_dataset(group, f'local_mass_density_{dens_type}kpccm', dtype='f8', units='Msun/kpccm**3')
                try:
                    # prop_dict[halo_type][f'local_mass_density_{dens_type}kpccm'].append(halo.local_mass_density[dens_type].in_units('Msun/kpccm**3'))
                    append_to_dataset(group, f'local_mass_density_{dens_type}kpccm', halo.local_mass_density[dens_type].in_units('Msun/kpccm**3'), curr_length)
                except:
                    print(f'Bad {halo_type} local_mass_density_{dens_type}kpccm')
                    # prop_dict[halo_type][f'local_mass_density_{dens_type}kpccm'].append(unyt.unyt_array(np.nan, 'Msun/kpccm**3'))
                    append_to_dataset(group, f'local_mass_density_{dens_type}kpccm', np.nan, curr_length)
    
                # first_time(prop_dict[halo_type], f'local_number_density_{dens_type}kpccm')
                create_dataset(group, f'local_number_density_{dens_type}kpccm', dtype='f8', units='kpccm**-3')
                try:
                    # prop_dict[halo_type][f'local_number_density_{dens_type}kpccm'].append(halo.local_number_density[dens_type].in_units('kpccm**-3'))
                    append_to_dataset(group, f'local_number_density_{dens_type}kpccm', halo.local_number_density[dens_type].in_units('kpccm**-3'), curr_length)
                except:
                    print(f'Bad {halo_type} local_number_density_{dens_type}kpccm')
                    # prop_dict[halo_type][f'local_v_density_{dens_type}kpccm'].append(unyt.unyt_array(np.nan, 'kpccm**-3'))
                    append_to_dataset(group, f'local_number_density_{dens_type}kpccm', np.nan, curr_length)


    
    
            ############# Check for nearby halos that qualify as major mergers ###########################
            
            # ## Make sphere around target halo (physical/proper units)
            # sphere_radius = sphere_radius_factor*halo.virial_quantities[sphere_radius_type].in_units(sphere_radius_units)
            # # sphere = target_snap.sphere(halo.minpotpos, sphere_radius)
        
            # ## Find all halos whose centres are within sphere
            # halo_minpotpos = unyt.unyt_array([_halo.minpotpos.in_units(sphere_radius_units) for _halo in target_obj.halos])
            # distance_from_target_halo = unyt.unyt_array(np.zeros(len(halo_minpotpos)), halo_minpotpos.units)
            # for ii in range(len(halo_minpotpos)):
            #     _minpotpos = halo_minpotpos[ii]
            #     distance_from_target_halo[ii] = euclidean_distance(_minpotpos, halo.minpotpos.in_units(sphere_radius_units))
            # halo_within_sphere_index = np.nonzero(distance_from_target_halo <= sphere_radius)[0]
            # halo_within_sphere_index = np.setdiff1d(halo_within_sphere_index, halo.GroupID) # Remove id of target halo
        
            # ## Check for major merger mass ratio
            # target_halo_m500c = halo.virial_quantities[mass_ratio_type]
            # halo_m500c = unyt.unyt_array([_halo.virial_quantities[mass_ratio_type] for _halo in target_obj.halos])
            # halos_within_sphere_m500c = halo_m500c[halo_within_sphere_index]
            # halo_m500c_ratio = []
            # for halo_within_sphere_m500c in halos_within_sphere_m500c:
            #     if target_halo_m500c >= halo_within_sphere_m500c:
            #         halo_m500c_ratio.append(target_halo_m500c/halo_within_sphere_m500c)
            #     else:
            #         halo_m500c_ratio.append(halo_within_sphere_m500c/target_halo_m500c)
            # halo_m500c_ratio = np.array(halo_m500c_ratio)
            # major_merger_indexes = np.where(halo_m500c_ratio <= major_merger_mass_ratio)[0]
            # # num_major_mergers = unyt.unyt_array(len(major_merger_indexes), '')
            # num_major_mergers = len(major_merger_indexes)
    
            # # first_time(prop_dict[halo_type], 'num_major_mergers')
            # create_dataset(group, 'num_major_mergers_phys', dtype='i8', units='1')
            # # prop_dict[halo_type]['num_major_mergers'].append(num_major_mergers)
            # append_to_dataset(group, 'num_major_mergers_phys', num_major_mergers, curr_length)
        
            # print()
        
            # ## dynamically deallocate variables to free up memory
            # del halo_m500c, halos_within_sphere_m500c, halo_m500c_ratio, major_merger_indexes #, sphere
            # gc.collect()



            ## Make sphere around target halo (comoving units)
            sphere_radius = sphere_radius_factor*halo.virial_quantities[sphere_radius_type].in_units(f'{sphere_radius_units}cm')
            # sphere = target_snap.sphere(halo.minpotpos, sphere_radius)
        
            ## Find all halos whose centres are within sphere
            halo_minpotpos = unyt.unyt_array([_halo.minpotpos.in_units(f'{sphere_radius_units}cm') for _halo in target_obj.halos])
            distance_from_target_halo = unyt.unyt_array(np.zeros(len(halo_minpotpos)), halo_minpotpos.units)
            for ii in range(len(halo_minpotpos)):
                _minpotpos = halo_minpotpos[ii]
                distance_from_target_halo[ii] = euclidean_distance(_minpotpos, halo.minpotpos.in_units(f'{sphere_radius_units}cm'))
            halo_within_sphere_index = np.nonzero(distance_from_target_halo <= sphere_radius)[0]
            halo_within_sphere_index = np.setdiff1d(halo_within_sphere_index, halo.GroupID) # Remove id of target halo
        
            ## Check for major merger mass ratio
            target_halo_m500c = halo.virial_quantities[mass_ratio_type]
            halo_m500c = unyt.unyt_array([_halo.virial_quantities[mass_ratio_type] for _halo in target_obj.halos])
            halos_within_sphere_m500c = halo_m500c[halo_within_sphere_index]
            halo_m500c_ratio = []
            for halo_within_sphere_m500c in halos_within_sphere_m500c:
                if target_halo_m500c >= halo_within_sphere_m500c:
                    halo_m500c_ratio.append(target_halo_m500c/halo_within_sphere_m500c)
                else:
                    halo_m500c_ratio.append(halo_within_sphere_m500c/target_halo_m500c)
            halo_m500c_ratio = np.array(halo_m500c_ratio)
            major_merger_indexes = np.where(halo_m500c_ratio <= major_merger_mass_ratio)[0]
            num_major_mergers = len(major_merger_indexes)
    
            # first_time(prop_dict[halo_type], 'num_major_mergers')
            create_dataset(group, 'num_major_mergers_cm', dtype='i8', units='1')
            # prop_dict[halo_type]['num_major_mergers'].append(num_major_mergers)
            append_to_dataset(group, 'num_major_mergers_cm', num_major_mergers, curr_length)
        
            print()
        
            ## dynamically deallocate variables to free up memory
            del halo_m500c, halos_within_sphere_m500c, halo_m500c_ratio, major_merger_indexes #, sphere
            gc.collect()
    
    
    

        print('\n\n\n')


    
        ## Central galaxy properties
        print('\n\nCalculating Halo Central Properties\n')

        # for central_type, central in zip(central_types, centrals):
        #     print(f'{central_type}: {central}\n')
        #     if central is None:
        #         continue
        if central is not None:
    
            central_type = 'central'
            # print(f'\n{central_type}\n')

            group = f[f'/{central_type}']
            # group = f['/central']
    
            # first_time(prop_dict[central_type], 'snap_num')
            create_dataset(group, 'snap_num', dtype='i8', units='1')
            curr_length = copy.deepcopy(len(group['snap_num'][:]))
            print()
            print(f"{central_type} snap_num = {group['snap_num'][:]}")
            print(f'curr_length = {curr_length}')
            print()
            if curr_length > 0:
                if target_snap_num in group['snap_num'][:-1]:
                    print(f'\nsnap_num {target_snap_num} for {central_type} already saved - skipping\n')
                    continue
                elif target_snap_num == group['snap_num'][:][-1]:
                    print(f'\nsnap_num {target_snap_num} for {central_type} likely only partly saved - overwriting properties\n')
                    curr_length -= 1
            # if target_snap_num in group['snap_num']:
            #     print(f'\nsnap_num {target_snap_num} for {central_type} already saved\n')
            #     continue
            # prop_dict[central_type]['snap_num'].append(unyt.unyt_array(target_snap_num, ''))
            append_to_dataset(group, 'snap_num', target_snap_num, curr_length)
    
            # first_time(prop_dict[central_type], 'age')
            create_dataset(group, 'age', dtype='f8', units='Gyr')
            # prop_dict[central_type]['age'].append(target_snap.current_time.in_units('Gyr'))
            append_to_dataset(group, 'age', target_snap.current_time.in_units('Gyr'), curr_length)
    
            # first_time(prop_dict[central_type], 'z')
            create_dataset(group, 'z', dtype='f8', units='1')
            # prop_dict[central_type]['z'].append(unyt.unyt_array(target_snap.current_redshift, ''))
            append_to_dataset(group, 'z', z_target, curr_length)
    
            # first_time(prop_dict[central_type], 'id')
            create_dataset(group, 'id', dtype='i8', units='1')
            # prop_dict[central_type]['id'].append(unyt.unyt_array(central.GroupID, ''))
            append_to_dataset(group, 'id', central.GroupID, curr_length)

            for ii, coord in zip([0,1,2], ['x', 'y', 'z']):
                for unit, suffix in zip(['', 'cm'], ['phys', 'cm']):
                    # create_dataset(group, f'minpotpos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
                    # append_to_dataset(group, f'minpotpos_{coord}_{suffix}',
                    #                   halo.minpotpos.in_units(f'Mpc{unit}')[ii], curr_length)

                    create_dataset(group, f'compos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
                    append_to_dataset(group, f'compos_{coord}_{suffix}',
                                      central.pos.in_units(f'Mpc{unit}')[ii], curr_length)
    
            # # first_time(prop_dict[central_type], 'minpotpos')
            # # create_dataset(group, 'minpotpos', dtype='f8')
            # create_dataset(group, 'minpotpos', shape=(0,3,), maxshape=(None,3,), dtype='f8')
            # # prop_dict[central_type]['minpotpos'].append(central.minpotpos.in_units('Mpccm'))
            # append_to_dataset(group, 'minpotpos', central.minpotpos.in_units('Mpccm'), curr_length)


            create_dataset(group, 'ngas', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'ngas', central.ngas, curr_length)
            except:
                print(f'Bad {central_type} ngas')
                append_to_dataset(group, 'ngas', 0., curr_length)

            create_dataset(group, 'nstar', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'nstar', central.nstar, curr_length)
            except:
                print(f'Bad {central_type} nstar')
                append_to_dataset(group, 'nstar', 0., curr_length)

            create_dataset(group, 'nbh', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'nbh', central.nbh, curr_length)
            except:
                print(f'Bad {central_type} nbh')
                append_to_dataset(group, 'nbh', 0., curr_length)

            create_dataset(group, 'ndm', dtype='f8', units='1')
            try:
                append_to_dataset(group, 'ndm', central.ndm, curr_length)
            except:
                print(f'Bad {central_type} ndm')
                append_to_dataset(group, 'ndm', 0., curr_length)




            ## Get centre of galaxy (pos) and bulk velocity of galaxy (comvel) from caesar
            ## for getting velocities relative to galaxy bulk velocity
            compos = central.pos
            comvel = central.vel



            aperture_names = ['3ckpc', '30ckpc', '50ckpc', 'r50star', 'r80tot']
            apertures = [target_snap.arr(3, 'kpccm'), target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm'), central.radii['stellar_half_mass'], central.radii['total_r80']]
            for aperture, aperture_name in zip(apertures, aperture_names):
                sphere = target_snap.sphere(compos, aperture)
                sphere.set_field_parameter('center', compos)
                sphere.set_field_parameter('bulk_velocity', comvel)

                for part_type in part_types:

                    create_dataset(group, f'n{part_type}_{aperture_name}', dtype='f8', units='1')
                    try:
                        npart = len(sphere[part_type, 'Masses'])
                        append_to_dataset(group, f'n{part_type}_{aperture_name}', npart, curr_length)
                    except:
                        print(f'Bad {central_type} n{part_type}_{aperture_name}')
                        append_to_dataset(group, f'n{part_type}_{aperture_name}', 0., curr_length)

                    create_dataset(group, f'm{part_type}_{aperture_name}', dtype='f8', units='Msun')
                    try:
                        mpart = sum(sphere[part_type, 'Masses'].in_units('Msun'))
                        append_to_dataset(group, f'm{part_type}_{aperture_name}', mpart, curr_length)
                    except:
                        print(f'Bad {central_type} m{part_type}_{aperture_name}')
                        append_to_dataset(group, f'm{part_type}_{aperture_name}', 0., curr_length)



                    if part_type == 'PartType0':
                        ## Cumulative number of wind launches
                        create_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', dtype='f8', units='1')
                        try:
                            nwind_launches = sum(sphere[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {central_type} {part_type}_NWindLaunches_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', dtype='f8', units='1')
                        try:
                            sphere_nowinds = sphere.cut_region([f"obj['{part_type}', 'DelayTime'].in_units('') == 0"])
                            nwind_launches = sum(sphere_nowinds[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {central_type} {part_type}_NWindLaunches_nowind_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_nowind_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', dtype='f8', units='1')
                        try:
                            sphere_windsonly = sphere.cut_region([f"obj['{part_type}', 'DelayTime'].in_units('') > 0"])
                            nwind_launches = sum(sphere_windsonly[part_type, 'NWindLaunches'])
                            append_to_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', nwind_launches, curr_length)
                        except:
                            print(f'Bad {central_type} {part_type}_NWindLaunches_onlywind_{aperture_name}')
                            append_to_dataset(group, f'{part_type}_NWindLaunches_onlywind_{aperture_name}', 0., curr_length)

                        
                        ## Mass of gas in different phases
                        for phase_def_name, phase_defs in gas_phases.items():
                            print(f'\nCalculating gas phases for definition: {phase_def_name}\n')
                            for phase_name, phase_criteria in phase_defs.items():
                                print(f'Phase: {phase_name}')
                                # phase_filter = np.ones(len(sphere[part_type, 'Masses']), dtype=bool)
                                phase_filter = []
                                for field_name, limits in phase_criteria.items():
                                    print(f'  Field: {field_name}')
                                    for limit_type, limit_value in limits.items():
                                        phase_filter.append(f"obj['{part_type}', '{field_name}'].in_units('{limit_value.units}') {limit_type} {limit_value.value}")

                                    # field_values = sphere[part_type, field_name]
                                    # if 'min' in limits:
                                    #     phase_filter &= (field_values >= limits['min'])
                                    # if 'max' in limits:
                                    #     phase_filter &= (field_values <= limits['max'])

                                dataset_name = f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'
                                create_dataset(group, dataset_name, dtype='f8', units='Msun')
                                try:
                                    sphere_phase = sphere.cut_region(phase_filter)
                                    mpart_phase = sum(sphere_phase[part_type, 'Masses'].in_units('Msun'))
                                    append_to_dataset(group, dataset_name, mpart_phase, curr_length)
                                except:
                                    print(f'Bad {central_type} {dataset_name}')
                                    append_to_dataset(group, dataset_name, 0., curr_length)

                                # try:
                                #     mpart_phase = sum(sphere[part_type, 'Masses'][phase_filter].in_units('Msun'))
                                #     append_to_dataset(group, dataset_name, mpart_phase, curr_length)
                                # except:
                                #     print(f'Bad {halo_type} {dataset_name}')
                                #     append_to_dataset(group, dataset_name, np.nan, curr_length)


                        ## Instantaneous star formation rate
                        create_dataset(group, f'sfr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
                        try:
                            sfr = sum(sphere[part_type, 'StarFormationRate'].in_units('Msun/yr'))
                            append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', sfr, curr_length)
                        except:
                            print(f'Bad {central_type} sfr_{part_type}_{aperture_name}')
                            append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', 0., curr_length)




                    if part_type == 'PartType4':
                        ## Time-averaged star formation rate
                        create_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
                        try:
                            # formation_times = sphere[part_type, 'StellarFormationTime']
                            ages = sphere[part_type, 'age']
                            # current_time = target_snap.current_time.in_units('Gyr')
                            # age_100Myr_filter = np.where((current_time - formation_times.in_units('Gyr')) <= 0.1)[0]
                            age_100Myr_filter = np.where(ages.in_units('Gyr') <= 0.1)[0]
                            masses = sphere[part_type, 'Masses'][age_100Myr_filter].in_units('Msun')
                            summed_mass = unyt.unyt_array(sum(masses), units='Msun')
                            time_diff = unyt.unyt_quantity(0.1, 'Gyr')
                            sfr_100Myr = summed_mass / time_diff.in_units('yr')
                            # sfr_100Myr = sum(sphere[part_type, 'Masses'][age_100Myr_filter].in_units('Msun')) / 0.1
                            append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', sfr_100Myr, curr_length)
                        except:
                            print(f'Bad {central_type} sfr_100Myr_{part_type}_{aperture_name}')
                            append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', 0., curr_length)
                        


                    
                    if part_type == 'PartType5':

                        create_dataset(group, f'm{part_type}_phys_{aperture_name}', dtype='f8', units='Msun')
                        try:
                            mpart = sum(sphere[part_type, 'BH_Mass'].in_units('Msun'))
                            append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', mpart, curr_length)
                        except:
                            print(f'Bad {central_type} m{part_type}_phys_{aperture_name}')
                            append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', 0., curr_length)


                        # create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')

                        # bh_properties_good = True
                        try:
                            ## Find Mbh,phys and f_edd for each BH
                            bh_mass = sphere[part_type, 'BH_Mass'].in_units('Msun')
                            bh_mdot = sphere[part_type, 'BH_Mdot'].in_units('Msun/yr')
                            # f_rad = 0.1
                            bh_mdot_acc = bh_mdot / (1 - f_rad) # Full accretion rate onto the BH not accounting for radiative losses
                            bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson)).in_units('Msun/yr')
                            bh_fedd = (bh_mdot/bh_mdot_edd).in_units('1')
                            print('\nbh_mass:')
                            print(bh_mass)
                            print()
                            print('\nbh_mdot:')
                            print(bh_mdot)
                            print()
                            print('\nbh_mdot_edd:')
                            print(bh_mdot_edd)
                            print()
                            print('\nbh_fedd:')
                            print(bh_fedd)
                            print()

                        except Exception as error:
                            print(f'Bad {central_type} {aperture_name} BH properties calculation')
                            print(error)
                            print()
                            try:
                                del bh_mass
                            except:
                                pass
                            try:
                                del bh_mdot
                            except:
                                pass
                            try:
                                del bh_mdot_acc
                            except:
                                pass
                            try:
                                del bh_mdot_edd
                            except:
                                pass
                            try:
                                del bh_fedd
                            except:
                                pass



                        create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', dtype='f8', units='1')
                        # create_dataset(group, f'n{part_type}_quasar_high_fedd_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')
                        create_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', dtype='f8', units='1')
                        # if not bh_properties_good:
                        try:
                            
                            # Find BHs with:
                            # No accretion (f_edd = 0);
                            # AGN in radiative/quasar mode (f_edd > 0.2);
                            # AGN in jet mode (f_edd <= 0.2) & Mbh,phys >= Mbh,jet,min (4e7 Msun for Simba, 7e7 Msun for Simba-C)
                            # (see e.g., Dave et al. 2019, Angles-Alcazar et al. 2020, Thomas et al. 2021, and references therein)
                            # bh_mass_jet_min = target_snap.arr(7e7, 'Msun')
                            # bh_fedd_jet_max = target_snap.arr(0.2, '1')

                            bh_no_accretion_filter = np.where(bh_fedd == unyt.unyt_quantity(0, ''))[0]

                            bh_quasar_filter = np.where(np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= bh_mass_jet_min))[0]
                            bh_quasar_ascale_filter = np.where(np.logical_and(bh_fedd > unyt.unyt_quantity(0, ''), bh_mass <= a_target*bh_mass_jet_min))[0]

                            bh_quasar_high_fedd_filter = np.where(bh_fedd >= bh_fedd_jet_max)[0]

                            bh_quasar_low_fedd_filter = np.where(np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]
                            bh_quasar_low_fedd_ascale_filter = np.where(np.logical_and(np.logical_and(bh_fedd < bh_fedd_jet_max, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]

                            bh_quasar_fedd002_filter = np.where(np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]
                            bh_quasar_fedd002_ascale_filter = np.where(np.logical_and(np.logical_and(bh_fedd < 0.02, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]
    
                            bh_quasar_fedd0002_filter = np.where(np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]
                            bh_quasar_fedd0002_ascale_filter = np.where(np.logical_and(np.logical_and(bh_fedd < 0.002, bh_mass <= a_target*bh_mass_jet_min), bh_fedd > unyt.unyt_quantity(0, '')))[0]

                            bh_jet_filter = np.where(np.logical_and(np.logical_and(bh_mass > bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, '')))[0]
                            bh_jet_ascale_filter = np.where(np.logical_and(np.logical_and(bh_mass > a_target*bh_mass_jet_min, bh_fedd < bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, '')))[0]


                            print('\nbh_no_accretion_filter:')
                            print(bh_no_accretion_filter)

                            print('\nbh_quasar_filter:')
                            print(bh_quasar_filter)
                            print('\nbh_quasar_ascale_filter:')
                            print(bh_quasar_ascale_filter)

                            print('\nbh_quasar_high_fedd_filter:')
                            print(bh_quasar_high_fedd_filter)

                            print('\nbh_quasar_low_fedd_filter:')
                            print(bh_quasar_low_fedd_filter)
                            print('\nbh_quasar_low_fedd_ascale_filter:')
                            print(bh_quasar_low_fedd_ascale_filter)

                            print('\nbh_quasar_fedd<0.02_filter:')
                            print(bh_quasar_fedd002_filter)
                            print('\nbh_quasar_fedd<0.02_ascale_filter:')
                            print(bh_quasar_fedd002_ascale_filter)

                            print('\nbh_quasar_fedd<0.002_filter:')
                            print(bh_quasar_fedd0002_filter)
                            print('\nbh_quasar_fedd<0.002_ascale_filter:')
                            print(bh_quasar_fedd0002_ascale_filter)

                            print('\nbh_jet_filter:')
                            print(bh_jet_filter)
                            print('\nbh_jet_ascale_filter:')
                            print(bh_jet_ascale_filter)
                            print()


                            nbh_no_accretion = len(bh_no_accretion_filter)

                            nbh_quasar = len(bh_quasar_filter)
                            nbh_quasar_ascale = len(bh_quasar_ascale_filter)

                            nbh_quasar_high_fedd = len(bh_quasar_high_fedd_filter)

                            nbh_quasar_low_fedd = len(bh_quasar_low_fedd_filter)
                            nbh_quasar_low_fedd_ascale = len(bh_quasar_low_fedd_ascale_filter)

                            nbh_quasar_fedd002 = len(bh_quasar_fedd002_filter)
                            nbh_quasar_fedd002_ascale = len(bh_quasar_fedd002_ascale_filter)

                            nbh_quasar_fedd0002 = len(bh_quasar_fedd0002_filter)
                            nbh_quasar_fedd0002_ascale = len(bh_quasar_fedd0002_ascale_filter)

                            nbh_jet = len(bh_jet_filter)
                            nbh_jet_ascale = len(bh_jet_ascale_filter)


                            append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', nbh_no_accretion, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', nbh_quasar, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', nbh_quasar_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', nbh_quasar_high_fedd, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', nbh_quasar_low_fedd, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', nbh_quasar_low_fedd_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', nbh_quasar_fedd002, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', nbh_quasar_fedd002_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', nbh_quasar_fedd0002, curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', nbh_quasar_fedd0002_ascale, curr_length)

                            append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', nbh_jet, curr_length)
                            append_to_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', nbh_jet_ascale, curr_length)

                        except Exception as error:
                            print(f'Bad {central_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
                            print(error)
                            append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_high_fedd_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_low_fedd_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.02_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_quasar_fedd<0.002_ascale_{aperture_name}', 0., curr_length)

                            append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', 0., curr_length)
                            append_to_dataset(group, f'n{part_type}_jets_ascale_{aperture_name}', 0., curr_length)



                        ## Calculate simple bolomentric AGN luminosity and jet power sums
                        create_dataset(group, f'bh_Lbol_{aperture_name}', dtype='f8', units='erg/s')
                        # create_dataset(group, f'Pjet_jets_{aperture_name}', dtype='f8', units='erg/s')
                        try:
                            Lbol = sum((f_rad * unyt.c**2 * bh_mdot).in_units('erg/s'))
                            append_to_dataset(group, f'bh_Lbol_{aperture_name}', Lbol, curr_length)
                        except:
                            print(f'Bad {central_type} bh_Lbol_{aperture_name}')
                            append_to_dataset(group, f'bh_Lbol_{aperture_name}', 0., curr_length)

                        create_dataset(group, f'bh_Lbol_acc_{aperture_name}', dtype='f8', units='erg/s')
                        try:
                            Lbol = sum((f_rad * unyt.c**2 * bh_mdot_acc).in_units('erg/s'))
                            append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', Lbol, curr_length)
                        except:
                            print(f'Bad {central_type} bh_Lbol_acc_{aperture_name}')
                            append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', 0., curr_length)






            # aperture_names = ['3ckpc', '30ckpc', '50ckpc']
            # apertures = [target_snap.arr(3, 'kpccm'), target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm')]
            # for aperture, aperture_name in zip(apertures, aperture_names):
            #     sphere = target_snap.sphere(central.pos, aperture)
            #     for part_type in part_types:
            #         create_dataset(group, f'n{part_type}_{aperture_name}', dtype='f8', units='1')
            #         try:
            #             npart = len(sphere[part_type, 'Masses'])
            #             append_to_dataset(group, f'n{part_type}_{aperture_name}', npart, curr_length)
            #         except:
            #             print(f'Bad {central_type} n{part_type}_{aperture_name}')
            #             append_to_dataset(group, f'n{part_type}_{aperture_name}', np.nan, curr_length)

            #         ## Number of BHs with active jets within aperture
            #         if part_type == 'PartType5':

            #             create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
            #             create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
            #             create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')
            #             try:
            #                 ## Find Mbh,phys and f_edd for each BH
            #                 bh_mass = sphere[part_type, 'BH_Mass'].in_units('Msun')
            #                 bh_mdot = sphere[part_type, 'BH_Mdot'].in_units('Msun/yr')
            #                 f_rad = 0.1
            #                 bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson)).in_units('Msun/yr')
            #                 bh_fedd = (bh_mdot/bh_mdot_edd).in_units('1')
            #                 print('\nbh_mass:')
            #                 print(bh_mass)
            #                 print()
            #                 print('\nbh_mdot:')
            #                 print(bh_mdot)
            #                 print()
            #                 print('\nbh_mdot_edd:')
            #                 print(bh_mdot_edd)
            #                 print()
            #                 print('\nbh_fedd:')
            #                 print(bh_fedd)
            #                 print()
                            
            #                 # Find BHs with:
            #                 # No accretion (f_edd = 0);
            #                 # AGN in radiative/quasar mode (f_edd > 0.2);
            #                 # AGN in jet mode (f_edd <= 0.2) & Mbh,phys >= Mbh,jet,min (4e7 Msun for Simba, 7e7 Msun for Simba-C)
            #                 # (see e.g., Dave et al. 2019, Angles-Alcazar et al. 2020, Thomas et al. 2021, and references therein)
            #                 bh_mass_jet_min = target_snap.arr(7e7, 'Msun')
            #                 bh_fedd_jet_max = target_snap.arr(0.2, '1')

            #                 bh_no_accretion_filter = np.where(bh_fedd == 0)[0]
            #                 bh_quasar_filter = np.where(bh_fedd > bh_fedd_jet_max)[0]
            #                 bh_jet_filter = np.where(np.logical_and(bh_mass >= bh_mass_jet_min, bh_fedd <= bh_fedd_jet_max))[0]
            #                 print('\nbh_no_accretion_filter:')
            #                 print(bh_no_accretion_filter)
            #                 print('\nbh_quasar_filter:')
            #                 print(bh_quasar_filter)
            #                 print('\nbh_jet_filter:')
            #                 print(bh_jet_filter)
            #                 print()

            #                 nbh_no_accretion = len(bh_no_accretion_filter)
            #                 nbh_quasar = len(bh_quasar_filter)
            #                 nbh_jet = len(bh_jet_filter)

            #                 append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', nbh_no_accretion, curr_length)
            #                 append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', nbh_quasar, curr_length)
            #                 append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', nbh_jet, curr_length)
            #             except Exception as error:
            #                 print(f'Bad {central_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
            #                 print(error)
            #                 print()
            #                 append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', np.nan, curr_length)
            #                 append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', np.nan, curr_length)
            #                 append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', np.nan, curr_length)
            
    
            # first_time(prop_dict[central_type], 'bh_mdot')
            create_dataset(group, 'bh_mdot', dtype='f8', units='Msun/yr')
            try:
                # prop_dict[central_type]['bh_mdot'].append(central.bhmdot.in_units('Msun/yr'))
                append_to_dataset(group, 'bh_mdot', central.bhmdot.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {central_type} bh_mdot')
                # prop_dict[central_type]['bh_mdot'].append(unyt.unyt_array(np.nan, 'Msun/yr'))
                append_to_dataset(group, 'bh_mdot', 0., curr_length)
    
            # first_time(prop_dict[central_type], 'bh_fedd')
            create_dataset(group, 'bh_fedd', dtype='f8', units='1')
            try:
                # prop_dict[central_type]['bh_fedd'].append(central.bh_fedd.in_units(''))
                append_to_dataset(group, 'bh_fedd', central.bh_fedd.in_units(''), curr_length)
            except:
                print(f'Bad {central_type} bh_fedd')
                # prop_dict[central_type]['bh_fedd'].append(unyt.unyt_array(np.nan, ''))
                append_to_dataset(group, 'bh_fedd', 0., curr_length)
    
            # first_time(prop_dict[central_type], 'bh_mdot_edd')
            # prop_dict[central_type]['bh_mdot_edd'].append(unyt.unyt_array(prop_dict[central_type]['bh_mdot'])/unyt.unyt_array(prop_dict[central_type]['bh_fedd']))
    
            # first_time(prop_dict[central_type], 'sfr')
            create_dataset(group, 'sfr', dtype='f8', units='Msun/yr')
            try:
                # prop_dict[central_type]['sfr'].append(central.sfr.in_units('Msun/yr'))
                append_to_dataset(group, 'sfr', central.sfr.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {central_type} sfr')
                # prop_dict[central_type]['sfr'].append(unyt.unyt_array(0, 'Msun/yr'))
                append_to_dataset(group, 'sfr', 0, curr_length)
    
            # first_time(prop_dict[central_type], 'sfr_100')
            create_dataset(group, 'sfr_100', dtype='f8', units='Msun/yr')
            try:
                # prop_dict[central_type]['sfr_100'].append(central.sfr_100.in_units('Msun/yr'))
                append_to_dataset(group, 'sfr_100', central.sfr_100.in_units('Msun/yr'), curr_length)
            except:
                print(f'Bad {central_type} sfr_100')
                # prop_dict[central_type]['sfr_100'].append(unyt.unyt_array(0, 'Msun/yr'))
                append_to_dataset(group, 'sfr_100', 0, curr_length)
    
            # for mass_type in central_mass_types:
            #     for aperture in central_mass_apertures:
            #         # if mass_type in ['stellar', 'dust'] and aperture=='_30kpc': continue
            #         # first_time(prop_dict[central_type], f'{mass_type}{aperture}_mass')
            #         create_dataset(group, f'{mass_type}{aperture}_mass', dtype='f8', units='Msun')
            #         try:
            #             # prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(central.masses[f'{mass_type}{aperture}'].in_units('Msun'))
            #             append_to_dataset(group, f'{mass_type}{aperture}_mass', central.masses[f'{mass_type}{aperture}'].in_units('Msun'), curr_length)
            #         except:
            #             print(f'Bad {central_type} {mass_type}{aperture}_mass')
            #             # prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(unyt.unyt_array(0, 'Msun'))
            #             append_to_dataset(group, f'{mass_type}{aperture}_mass', 0*units.Msun, curr_length)

            for mass_type in central_mass_types:
                create_dataset(group, f'{mass_type}_mass', dtype='f8', units='Msun')
                try:
                    append_to_dataset(group, f'{mass_type}_mass', central.masses[f'{mass_type}'].in_units('Msun'), curr_length)
                except:
                    print(f'Bad {central_type} {mass_type}_mass')
                    append_to_dataset(group, f'{mass_type}_mass', 0, curr_length)
    
            for radii_type in central_radii_types:
                for XX in central_radii_XX:
                    # first_time(prop_dict[central_type], f'{radii_type}_{XX}_radius')
                    create_dataset(group, f'{radii_type}_{XX}_radius', dtype='f8', units='kpc')
                    try:
                        # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(central.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius', central.radii[f'{radii_type}_{XX}'].in_units('kpc'), curr_length)
                    except:
                        print(f'Bad {central_type} {radii_type}_{XX}_radius')
                        # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius', np.nan, curr_length)

                    create_dataset(group, f'{radii_type}_{XX}_radius_cm', dtype='f8', units='kpccm')
                    try:
                        # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(central.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', central.radii[f'{radii_type}_{XX}'].in_units('kpccm'), curr_length)
                    except:
                        print(f'Bad {central_type} {radii_type}_{XX}_radius_cm')
                        # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
                        append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', np.nan, curr_length)
    
            for metal_type in central_metallicity_types:
                # first_time(prop_dict[central_type], f'{metal_type}_metallicity')
                create_dataset(group, f'{metal_type}_metallicity', dtype='f8', units='1')
                try:
                    # prop_dict[central_type][f'{metal_type}_metallicity'].append(central.metallicities[metal_type])
                    append_to_dataset(group, f'{metal_type}_metallicity', central.metallicities[metal_type], curr_length)
                except:
                    print(f'Bad {central_type} {metal_type}_metallicity')
                    # prop_dict[central_type][f'{metal_type}_metallicity'].append(unyt.unyt_array(np.nan, ''))
                    append_to_dataset(group, f'{metal_type}_metallicity', np.nan, curr_length)
    
            for vel_disp_type in central_velocity_dispersion_types:
                # first_time(prop_dict[central_type], f'{vel_disp_type}_velocity_dispersion')
                create_dataset(group, f'{vel_disp_type}_velocity_dispersion', dtype='f8', units='km/s')
                try:
                    # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(central.velocity_dispersions[vel_disp_type].in_units('km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', central.velocity_dispersions[vel_disp_type].in_units('km/s'), curr_length)
                except:
                    print(f'Bad {central_type} {vel_disp_type}_velocity_dispersion')
                    # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', np.nan, curr_length)

                create_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', dtype='f8', units='kmcm/s')
                try:
                    # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(central.velocity_dispersions[vel_disp_type].in_units('km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', central.velocity_dispersions[vel_disp_type].in_units('kmcm/s'), curr_length)
                except:
                    print(f'Bad {central_type} {vel_disp_type}_velocity_dispersion_cm')
                    # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
                    append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', np.nan, curr_length)
    
            for age_type in central_age_types:
                # first_time(prop_dict[central_type], f'{age_type}_stellar_age')
                create_dataset(group, f'{age_type}_stellar_age', dtype='f8', units='Gyr')
                try:
                    # prop_dict[central_type][f'{age_type}_stellar_age'].append(central.ages[age_type].in_units('Gyr'))
                    append_to_dataset(group, f'{age_type}_stellar_age', central.ages[age_type].in_units('Gyr'), curr_length)
                except:
                    print(f'Bad {central_type} {age_type}_stellar_age')
                    # prop_dict[central_type][f'{age_type}_stellar_age'].append(unyt.unyt_array(np.nan, 'Gyr'))
                    append_to_dataset(group, f'{age_type}_stellar_age', np.nan, curr_length)
    
            for temp_type in central_temperature_types:
                # first_time(prop_dict[central_type], f'{temp_type}_temperature')
                create_dataset(group, f'{temp_type}_temperature', dtype='f8', units='K')
                try:
                    # prop_dict[central_type][f'{temp_type}_temperature'].append(central.temperatures[temp_type].in_units('K'))
                    append_to_dataset(group, f'{temp_type}_temperature', central.temperatures[temp_type].in_units('K'), curr_length)
                except:
                    print(f'Bad {central_type} {temp_type}_temperature')
                    # prop_dict[central_type][f'{temp_type}_temperature'].append(unyt.unyt_array(np.nan, 'K'))
                    append_to_dataset(group, f'{temp_type}_temperature', np.nan, curr_length)

            print()





        # ## Satellite galaxy properties
        # print('\n\nCalculating Halo Satellite Properties\n')
        # if len(satellites) > 0:
    
        #     central_type = 'satellite'
        #     # print(f'\n{central_type}\n')

        #     group = f[f'/{central_type}']
        #     # group = f['/central']
    
        #     # first_time(prop_dict[central_type], 'snap_num')
        #     create_dataset(group, 'snap_num', dtype='i8', units='1')
        #     curr_length = copy.deepcopy(len(group['snap_num'][:]))
        #     print()
        #     print(f"{central_type} snap_num = {group['snap_num'][:]}")
        #     print(f'curr_length = {curr_length}')
        #     print()
        #     if curr_length > 0:
        #         if target_snap_num in group['snap_num'][:-1]:
        #             print(f'\nsnap_num {target_snap_num} for {central_type} already saved - skipping\n')
        #             continue
        #         elif target_snap_num == group['snap_num'][:][-1]:
        #             print(f'\nsnap_num {target_snap_num} for {central_type} likely only partly saved - overwriting properties\n')
        #             curr_length -= 1
        #     # if target_snap_num in group['snap_num']:
        #     #     print(f'\nsnap_num {target_snap_num} for {central_type} already saved\n')
        #     #     continue
        #     # prop_dict[central_type]['snap_num'].append(unyt.unyt_array(target_snap_num, ''))
        #     append_to_dataset(group, 'snap_num', target_snap_num, curr_length)
    
        #     # first_time(prop_dict[central_type], 'age')
        #     create_dataset(group, 'age', dtype='f8', units='Gyr')
        #     # prop_dict[central_type]['age'].append(target_snap.current_time.in_units('Gyr'))
        #     append_to_dataset(group, 'age', target_snap.current_time.in_units('Gyr'), curr_length)
    
        #     # first_time(prop_dict[central_type], 'z')
        #     create_dataset(group, 'z', dtype='f8', units='1')
        #     # prop_dict[central_type]['z'].append(unyt.unyt_array(target_snap.current_redshift, ''))
        #     append_to_dataset(group, 'z', target_snap.current_redshift, curr_length)

        #     create_dataset(group, 'n_sat', dtype='f8', units='1')
        #     append_to_dataset(group, 'n_sat', len(satellites), curr_length)
    
        #     # # first_time(prop_dict[central_type], 'id')
        #     # create_dataset(group, 'id', dtype='i8', units='1')
        #     # # prop_dict[central_type]['id'].append(unyt.unyt_array(central.GroupID, ''))
        #     # append_to_dataset(group, 'id', central.GroupID, curr_length)

        #     # for ii, coord in zip([0,1,2], ['x', 'y', 'z']):
        #     #     for unit, suffix in zip(['', 'cm'], ['phys', 'cm']):
        #     #         # create_dataset(group, f'minpotpos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
        #     #         # append_to_dataset(group, f'minpotpos_{coord}_{suffix}',
        #     #         #                   halo.minpotpos.in_units(f'Mpc{unit}')[ii], curr_length)

        #     #         create_dataset(group, f'compos_{coord}_{suffix}', dtype='f8', units=f'Mpc{unit}')
        #     #         append_to_dataset(group, f'compos_{coord}_{suffix}',
        #     #                           central.pos.in_units(f'Mpc{unit}')[ii], curr_length)
    
        #     # # first_time(prop_dict[central_type], 'minpotpos')
        #     # # create_dataset(group, 'minpotpos', dtype='f8')
        #     # create_dataset(group, 'minpotpos', shape=(0,3,), maxshape=(None,3,), dtype='f8')
        #     # # prop_dict[central_type]['minpotpos'].append(central.minpotpos.in_units('Mpccm'))
        #     # append_to_dataset(group, 'minpotpos', central.minpotpos.in_units('Mpccm'), curr_length)


        #     create_dataset(group, 'ngas', dtype='f8', units='1')
        #     try:
        #         value = sum(np.array([sat.ngas for sat in satellites]))
        #         append_to_dataset(group, 'ngas', value, curr_length)
        #     except:
        #         print(f'Bad {central_type} ngas')
        #         append_to_dataset(group, 'ngas', 0., curr_length)

        #     create_dataset(group, 'nstar', dtype='f8', units='1')
        #     try:
        #         value = sum(np.array([sat.nstar for sat in satellites]))
        #         append_to_dataset(group, 'nstar', value, curr_length)
        #     except:
        #         print(f'Bad {central_type} nstar')
        #         append_to_dataset(group, 'nstar', 0., curr_length)

        #     create_dataset(group, 'nbh', dtype='f8', units='1')
        #     try:
        #         value = sum(np.array([sat.nbh for sat in satellites]))
        #         append_to_dataset(group, 'nbh', value, curr_length)
        #     except:
        #         print(f'Bad {central_type} nbh')
        #         append_to_dataset(group, 'nbh', 0., curr_length)

        #     create_dataset(group, 'ndm', dtype='f8', units='1')
        #     try:
        #         value = sum(np.array([sat.ndm for sat in satellites]))
        #         append_to_dataset(group, 'ndm', value curr_length)
        #     except:
        #         print(f'Bad {central_type} ndm')
        #         append_to_dataset(group, 'ndm', 0., curr_length)




        #     ## Get centre of galaxy (pos) and bulk velocity of galaxy (comvel) from caesar
        #     ## for getting velocities relative to galaxy bulk velocity
        #     pos = halo.minpotpos
        #     vel = halo.vel



        #     aperture_names = ['3ckpc', '30ckpc', '50ckpc']
        #     apertures = [target_snap.arr(3, 'kpccm'), target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm')]
        #     for aperture, aperture_name in zip(apertures, aperture_names):
        #         sphere = target_snap.sphere(compos, aperture)
        #         sphere.set_field_parameter('center', compos)
        #         sphere.set_field_parameter('bulk_velocity', comvel)

        #         for part_type in part_types:

        #             create_dataset(group, f'n{part_type}_{aperture_name}', dtype='f8', units='1')
        #             try:
        #                 npart = len(sphere[part_type, 'Masses'])
        #                 append_to_dataset(group, f'n{part_type}_{aperture_name}', npart, curr_length)
        #             except:
        #                 print(f'Bad {central_type} n{part_type}_{aperture_name}')
        #                 append_to_dataset(group, f'n{part_type}_{aperture_name}', 0., curr_length)

        #             create_dataset(group, f'm{part_type}_{aperture_name}', dtype='f8', units='Msun')
        #             try:
        #                 mpart = sum(sphere[part_type, 'Masses'].in_units('Msun'))
        #                 append_to_dataset(group, f'm{part_type}_{aperture_name}', mpart, curr_length)
        #             except:
        #                 print(f'Bad {central_type} m{part_type}_{aperture_name}')
        #                 append_to_dataset(group, f'm{part_type}_{aperture_name}', 0., curr_length)



        #             if part_type == 'PartType0':
        #                 ## Cumulative number of wind launches
        #                 create_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', dtype='f8', units='1')
        #                 try:
        #                     nwind_launches = sum(sphere[part_type, 'NWindLaunches'])
        #                     append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', nwind_launches, curr_length)
        #                 except:
        #                     print(f'Bad {halo_type} {part_type}_NWindLaunches_{aperture_name}')
        #                     append_to_dataset(group, f'{part_type}_NWindLaunches_{aperture_name}', 0., curr_length)

                        
        #                 ## Mass of gas in different phases
        #                 for phase_def_name, phase_defs in gas_phases.items():
        #                     print(f'\nCalculating gas phases for definition: {phase_def_name}\n')
        #                     for phase_name, phase_criteria in phase_defs.items():
        #                         print(f'Phase: {phase_name}')
        #                         # phase_filter = np.ones(len(sphere[part_type, 'Masses']), dtype=bool)
        #                         phase_filter = []
        #                         for field_name, limits in phase_criteria.items():
        #                             print(f'  Field: {field_name}')
        #                             for limit_type, limit_value in limits.items():
        #                                 phase_filter.append(f"obj['{part_type}', '{field_name}'].in_units('{limit_value.units}') {limit_type} {limit_value.value}")

        #                             # field_values = sphere[part_type, field_name]
        #                             # if 'min' in limits:
        #                             #     phase_filter &= (field_values >= limits['min'])
        #                             # if 'max' in limits:
        #                             #     phase_filter &= (field_values <= limits['max'])

        #                         dataset_name = f'm{part_type}_{phase_def_name}_{phase_name}_{aperture_name}'
        #                         create_dataset(group, dataset_name, dtype='f8', units='Msun')
        #                         try:
        #                             sphere_phase = sphere.cut_region(phase_filter)
        #                             mpart_phase = sum(sphere_phase[part_type, 'Masses'].in_units('Msun'))
        #                             append_to_dataset(group, dataset_name, mpart_phase, curr_length)
        #                         except:
        #                             print(f'Bad {central_type} {dataset_name}')
        #                             append_to_dataset(group, dataset_name, 0., curr_length)

        #                         # try:
        #                         #     mpart_phase = sum(sphere[part_type, 'Masses'][phase_filter].in_units('Msun'))
        #                         #     append_to_dataset(group, dataset_name, mpart_phase, curr_length)
        #                         # except:
        #                         #     print(f'Bad {halo_type} {dataset_name}')
        #                         #     append_to_dataset(group, dataset_name, np.nan, curr_length)


        #                 ## Instantaneous star formation rate
        #                 create_dataset(group, f'sfr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
        #                 try:
        #                     sfr = sum(sphere[part_type, 'StarFormationRate'].in_units('Msun/yr'))
        #                     append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', sfr, curr_length)
        #                 except:
        #                     print(f'Bad {central_type} sfr_{part_type}_{aperture_name}')
        #                     append_to_dataset(group, f'sfr_{part_type}_{aperture_name}', 0., curr_length)




        #             if part_type == 'PartType4':
        #                 ## Time-averaged star formation rate
        #                 create_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', dtype='f8', units='Msun/yr')
        #                 try:
        #                     formation_times = sphere[part_type, 'StellarFormationTime']
        #                     current_time = target_snap.current_time.in_units('Gyr')
        #                     age_100Myr_filter = np.where((current_time - formation_times.in_units('Gyr')) <= 0.1)[0]
        #                     sfr_100Myr = sum(sphere[part_type, 'Masses'][age_100Myr_filter].in_units('Msun')) / 0.1
        #                     append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', sfr_100Myr, curr_length)
        #                 except:
        #                     print(f'Bad {central_type} sfr_100Myr_{part_type}_{aperture_name}')
        #                     append_to_dataset(group, f'sfr_100Myr_{part_type}_{aperture_name}', 0., curr_length)
                        


                    
        #             if part_type == 'PartType5':

        #                 create_dataset(group, f'm{part_type}_phys_{aperture_name}', dtype='f8', units='Msun')
        #                 try:
        #                     mpart = sum(sphere[part_type, 'BH_Mass'].in_units('Msun'))
        #                     append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', mpart, curr_length)
        #                 except:
        #                     print(f'Bad {central_type} m{part_type}_phys_{aperture_name}')
        #                     append_to_dataset(group, f'm{part_type}_phys_{aperture_name}', 0., curr_length)


        #                 # create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
        #                 # create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
        #                 # create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')

        #                 # bh_properties_good = True
        #                 try:
        #                     ## Find Mbh,phys and f_edd for each BH
        #                     bh_mass = sphere[part_type, 'BH_Mass'].in_units('Msun')
        #                     bh_mdot = sphere[part_type, 'BH_Mdot'].in_units('Msun/yr')
        #                     # f_rad = 0.1
        #                     bh_mdot_acc = bh_mdot / (1 - f_rad) # Full accretion rate onto the BH not accounting for radiative losses
        #                     bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson)).in_units('Msun/yr')
        #                     bh_fedd = (bh_mdot/bh_mdot_edd).in_units('1')
        #                     print('\nbh_mass:')
        #                     print(bh_mass)
        #                     print()
        #                     print('\nbh_mdot:')
        #                     print(bh_mdot)
        #                     print()
        #                     print('\nbh_mdot_edd:')
        #                     print(bh_mdot_edd)
        #                     print()
        #                     print('\nbh_fedd:')
        #                     print(bh_fedd)
        #                     print()

        #                 except Exception as error:
        #                     print(f'Bad {central_type} {aperture_name} BH properties calculation')
        #                     # bh_properties_good = False
        #                     print(error)
        #                     print()



        #                 create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
        #                 create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
        #                 create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')
        #                 # if not bh_properties_good:
        #                 try:
                            
        #                     # Find BHs with:
        #                     # No accretion (f_edd = 0);
        #                     # AGN in radiative/quasar mode (f_edd > 0.2);
        #                     # AGN in jet mode (f_edd <= 0.2) & Mbh,phys >= Mbh,jet,min (4e7 Msun for Simba, 7e7 Msun for Simba-C)
        #                     # (see e.g., Dave et al. 2019, Angles-Alcazar et al. 2020, Thomas et al. 2021, and references therein)
        #                     # bh_mass_jet_min = target_snap.arr(7e7, 'Msun')
        #                     # bh_fedd_jet_max = target_snap.arr(0.2, '1')

        #                     bh_no_accretion_filter = np.where(bh_fedd == unyt.unyt_quantity(0, ''))[0]
        #                     bh_quasar_filter = np.where(bh_fedd > bh_fedd_jet_max)[0]
        #                     bh_jet_filter = np.where(np.logical_and(np.logical_and(bh_mass >= bh_mass_jet_min, bh_fedd <= bh_fedd_jet_max), bh_fedd > unyt.unyt_quantity(0, '')))[0]
        #                     print('\nbh_no_accretion_filter:')
        #                     print(bh_no_accretion_filter)
        #                     print('\nbh_quasar_filter:')
        #                     print(bh_quasar_filter)
        #                     print('\nbh_jet_filter:')
        #                     print(bh_jet_filter)
        #                     print()

        #                     nbh_no_accretion = len(bh_no_accretion_filter)
        #                     nbh_quasar = len(bh_quasar_filter)
        #                     nbh_jet = len(bh_jet_filter)

        #                     append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', nbh_no_accretion, curr_length)
        #                     append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', nbh_quasar, curr_length)
        #                     append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', nbh_jet, curr_length)

        #                 except:
        #                     print(f'Bad {central_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
        #                     append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', 0., curr_length)
        #                     append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', 0., curr_length)
        #                     append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', 0., curr_length)



        #                 ## Calculate simple bolomentric AGN luminosity and jet power sums
        #                 create_dataset(group, f'bh_Lbol_{aperture_name}', dtype='f8', units='erg/s')
        #                 # create_dataset(group, f'Pjet_jets_{aperture_name}', dtype='f8', units='erg/s')
        #                 try:
        #                     Lbol = sum((f_rad * unyt.c**2 * bh_mdot).in_units('erg/s'))
        #                     append_to_dataset(group, f'bh_Lbol_{aperture_name}', Lbol, curr_length)
        #                 except:
        #                     print(f'Bad {central_type} bh_Lbol_{aperture_name}')
        #                     append_to_dataset(group, f'bh_Lbol_{aperture_name}', 0., curr_length)

        #                 create_dataset(group, f'bh_Lbol_acc_{aperture_name}', dtype='f8', units='erg/s')
        #                 try:
        #                     Lbol = sum((f_rad * unyt.c**2 * bh_mdot_acc).in_units('erg/s'))
        #                     append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', Lbol, curr_length)
        #                 except:
        #                     print(f'Bad {central_type} bh_Lbol_acc_{aperture_name}')
        #                     append_to_dataset(group, f'bh_Lbol_acc_{aperture_name}', 0., curr_length)






        #     # aperture_names = ['3ckpc', '30ckpc', '50ckpc']
        #     # apertures = [target_snap.arr(3, 'kpccm'), target_snap.arr(30, 'kpccm'), target_snap.arr(50, 'kpccm')]
        #     # for aperture, aperture_name in zip(apertures, aperture_names):
        #     #     sphere = target_snap.sphere(central.pos, aperture)
        #     #     for part_type in part_types:
        #     #         create_dataset(group, f'n{part_type}_{aperture_name}', dtype='f8', units='1')
        #     #         try:
        #     #             npart = len(sphere[part_type, 'Masses'])
        #     #             append_to_dataset(group, f'n{part_type}_{aperture_name}', npart, curr_length)
        #     #         except:
        #     #             print(f'Bad {central_type} n{part_type}_{aperture_name}')
        #     #             append_to_dataset(group, f'n{part_type}_{aperture_name}', np.nan, curr_length)

        #     #         ## Number of BHs with active jets within aperture
        #     #         if part_type == 'PartType5':

        #     #             create_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', dtype='f8', units='1')
        #     #             create_dataset(group, f'n{part_type}_quasar_{aperture_name}', dtype='f8', units='1')
        #     #             create_dataset(group, f'n{part_type}_jets_{aperture_name}', dtype='f8', units='1')
        #     #             try:
        #     #                 ## Find Mbh,phys and f_edd for each BH
        #     #                 bh_mass = sphere[part_type, 'BH_Mass'].in_units('Msun')
        #     #                 bh_mdot = sphere[part_type, 'BH_Mdot'].in_units('Msun/yr')
        #     #                 f_rad = 0.1
        #     #                 bh_mdot_edd = (4*np.pi*unyt.G*unyt.mp*bh_mass / (f_rad*unyt.c*unyt.sigma_thomson)).in_units('Msun/yr')
        #     #                 bh_fedd = (bh_mdot/bh_mdot_edd).in_units('1')
        #     #                 print('\nbh_mass:')
        #     #                 print(bh_mass)
        #     #                 print()
        #     #                 print('\nbh_mdot:')
        #     #                 print(bh_mdot)
        #     #                 print()
        #     #                 print('\nbh_mdot_edd:')
        #     #                 print(bh_mdot_edd)
        #     #                 print()
        #     #                 print('\nbh_fedd:')
        #     #                 print(bh_fedd)
        #     #                 print()
                            
        #     #                 # Find BHs with:
        #     #                 # No accretion (f_edd = 0);
        #     #                 # AGN in radiative/quasar mode (f_edd > 0.2);
        #     #                 # AGN in jet mode (f_edd <= 0.2) & Mbh,phys >= Mbh,jet,min (4e7 Msun for Simba, 7e7 Msun for Simba-C)
        #     #                 # (see e.g., Dave et al. 2019, Angles-Alcazar et al. 2020, Thomas et al. 2021, and references therein)
        #     #                 bh_mass_jet_min = target_snap.arr(7e7, 'Msun')
        #     #                 bh_fedd_jet_max = target_snap.arr(0.2, '1')

        #     #                 bh_no_accretion_filter = np.where(bh_fedd == 0)[0]
        #     #                 bh_quasar_filter = np.where(bh_fedd > bh_fedd_jet_max)[0]
        #     #                 bh_jet_filter = np.where(np.logical_and(bh_mass >= bh_mass_jet_min, bh_fedd <= bh_fedd_jet_max))[0]
        #     #                 print('\nbh_no_accretion_filter:')
        #     #                 print(bh_no_accretion_filter)
        #     #                 print('\nbh_quasar_filter:')
        #     #                 print(bh_quasar_filter)
        #     #                 print('\nbh_jet_filter:')
        #     #                 print(bh_jet_filter)
        #     #                 print()

        #     #                 nbh_no_accretion = len(bh_no_accretion_filter)
        #     #                 nbh_quasar = len(bh_quasar_filter)
        #     #                 nbh_jet = len(bh_jet_filter)

        #     #                 append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', nbh_no_accretion, curr_length)
        #     #                 append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', nbh_quasar, curr_length)
        #     #                 append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', nbh_jet, curr_length)
        #     #             except Exception as error:
        #     #                 print(f'Bad {central_type} n{part_type}_no_accretion/quasar/jets_{aperture_name}')
        #     #                 print(error)
        #     #                 print()
        #     #                 append_to_dataset(group, f'n{part_type}_no_accretion_{aperture_name}', np.nan, curr_length)
        #     #                 append_to_dataset(group, f'n{part_type}_quasar_{aperture_name}', np.nan, curr_length)
        #     #                 append_to_dataset(group, f'n{part_type}_jets_{aperture_name}', np.nan, curr_length)
            
    
        #     # first_time(prop_dict[central_type], 'bh_mdot')
        #     create_dataset(group, 'bh_mdot', dtype='f8', units='Msun/yr')
        #     try:
        #         # prop_dict[central_type]['bh_mdot'].append(central.bhmdot.in_units('Msun/yr'))
        #         append_to_dataset(group, 'bh_mdot', central.bhmdot.in_units('Msun/yr'), curr_length)
        #     except:
        #         print(f'Bad {central_type} bh_mdot')
        #         # prop_dict[central_type]['bh_mdot'].append(unyt.unyt_array(np.nan, 'Msun/yr'))
        #         append_to_dataset(group, 'bh_mdot', 0., curr_length)
    
        #     # first_time(prop_dict[central_type], 'bh_fedd')
        #     create_dataset(group, 'bh_fedd', dtype='f8', units='1')
        #     try:
        #         # prop_dict[central_type]['bh_fedd'].append(central.bh_fedd.in_units(''))
        #         append_to_dataset(group, 'bh_fedd', central.bh_fedd.in_units(''), curr_length)
        #     except:
        #         print(f'Bad {central_type} bh_fedd')
        #         # prop_dict[central_type]['bh_fedd'].append(unyt.unyt_array(np.nan, ''))
        #         append_to_dataset(group, 'bh_fedd', 0., curr_length)
    
        #     # first_time(prop_dict[central_type], 'bh_mdot_edd')
        #     # prop_dict[central_type]['bh_mdot_edd'].append(unyt.unyt_array(prop_dict[central_type]['bh_mdot'])/unyt.unyt_array(prop_dict[central_type]['bh_fedd']))
    
        #     # first_time(prop_dict[central_type], 'sfr')
        #     create_dataset(group, 'sfr', dtype='f8', units='Msun/yr')
        #     try:
        #         # prop_dict[central_type]['sfr'].append(central.sfr.in_units('Msun/yr'))
        #         append_to_dataset(group, 'sfr', central.sfr.in_units('Msun/yr'), curr_length)
        #     except:
        #         print(f'Bad {central_type} sfr')
        #         # prop_dict[central_type]['sfr'].append(unyt.unyt_array(0, 'Msun/yr'))
        #         append_to_dataset(group, 'sfr', 0, curr_length)
    
        #     # first_time(prop_dict[central_type], 'sfr_100')
        #     create_dataset(group, 'sfr_100', dtype='f8', units='Msun/yr')
        #     try:
        #         # prop_dict[central_type]['sfr_100'].append(central.sfr_100.in_units('Msun/yr'))
        #         append_to_dataset(group, 'sfr_100', central.sfr_100.in_units('Msun/yr'), curr_length)
        #     except:
        #         print(f'Bad {central_type} sfr_100')
        #         # prop_dict[central_type]['sfr_100'].append(unyt.unyt_array(0, 'Msun/yr'))
        #         append_to_dataset(group, 'sfr_100', 0, curr_length)
    
        #     # for mass_type in central_mass_types:
        #     #     for aperture in central_mass_apertures:
        #     #         # if mass_type in ['stellar', 'dust'] and aperture=='_30kpc': continue
        #     #         # first_time(prop_dict[central_type], f'{mass_type}{aperture}_mass')
        #     #         create_dataset(group, f'{mass_type}{aperture}_mass', dtype='f8', units='Msun')
        #     #         try:
        #     #             # prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(central.masses[f'{mass_type}{aperture}'].in_units('Msun'))
        #     #             append_to_dataset(group, f'{mass_type}{aperture}_mass', central.masses[f'{mass_type}{aperture}'].in_units('Msun'), curr_length)
        #     #         except:
        #     #             print(f'Bad {central_type} {mass_type}{aperture}_mass')
        #     #             # prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(unyt.unyt_array(0, 'Msun'))
        #     #             append_to_dataset(group, f'{mass_type}{aperture}_mass', 0*units.Msun, curr_length)

        #     for mass_type in central_mass_types:
        #         create_dataset(group, f'{mass_type}_mass', dtype='f8', units='Msun')
        #         try:
        #             append_to_dataset(group, f'{mass_type}_mass', central.masses[f'{mass_type}'].in_units('Msun'), curr_length)
        #         except:
        #             print(f'Bad {central_type} {mass_type}_mass')
        #             append_to_dataset(group, f'{mass_type}_mass', 0, curr_length)
    
        #     for radii_type in central_radii_types:
        #         for XX in central_radii_XX:
        #             # first_time(prop_dict[central_type], f'{radii_type}_{XX}_radius')
        #             create_dataset(group, f'{radii_type}_{XX}_radius', dtype='f8', units='kpc')
        #             try:
        #                 # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(central.radii[f'{radii_type}_{XX}'].in_units('kpc'))
        #                 append_to_dataset(group, f'{radii_type}_{XX}_radius', central.radii[f'{radii_type}_{XX}'].in_units('kpc'), curr_length)
        #             except:
        #                 print(f'Bad {central_type} {radii_type}_{XX}_radius')
        #                 # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
        #                 append_to_dataset(group, f'{radii_type}_{XX}_radius', np.nan, curr_length)

        #             create_dataset(group, f'{radii_type}_{XX}_radius_cm', dtype='f8', units='kpccm')
        #             try:
        #                 # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(central.radii[f'{radii_type}_{XX}'].in_units('kpc'))
        #                 append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', central.radii[f'{radii_type}_{XX}'].in_units('kpccm'), curr_length)
        #             except:
        #                 print(f'Bad {central_type} {radii_type}_{XX}_radius_cm')
        #                 # prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))
        #                 append_to_dataset(group, f'{radii_type}_{XX}_radius_cm', np.nan, curr_length)
    
        #     for metal_type in central_metallicity_types:
        #         # first_time(prop_dict[central_type], f'{metal_type}_metallicity')
        #         create_dataset(group, f'{metal_type}_metallicity', dtype='f8', units='1')
        #         try:
        #             # prop_dict[central_type][f'{metal_type}_metallicity'].append(central.metallicities[metal_type])
        #             append_to_dataset(group, f'{metal_type}_metallicity', central.metallicities[metal_type], curr_length)
        #         except:
        #             print(f'Bad {central_type} {metal_type}_metallicity')
        #             # prop_dict[central_type][f'{metal_type}_metallicity'].append(unyt.unyt_array(np.nan, ''))
        #             append_to_dataset(group, f'{metal_type}_metallicity', np.nan, curr_length)
    
        #     for vel_disp_type in central_velocity_dispersion_types:
        #         # first_time(prop_dict[central_type], f'{vel_disp_type}_velocity_dispersion')
        #         create_dataset(group, f'{vel_disp_type}_velocity_dispersion', dtype='f8', units='km/s')
        #         try:
        #             # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(central.velocity_dispersions[vel_disp_type].in_units('km/s'))
        #             append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', central.velocity_dispersions[vel_disp_type].in_units('km/s'), curr_length)
        #         except:
        #             print(f'Bad {central_type} {vel_disp_type}_velocity_dispersion')
        #             # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
        #             append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion', np.nan, curr_length)

        #         create_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', dtype='f8', units='kmcm/s')
        #         try:
        #             # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(central.velocity_dispersions[vel_disp_type].in_units('km/s'))
        #             append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', central.velocity_dispersions[vel_disp_type].in_units('kmcm/s'), curr_length)
        #         except:
        #             print(f'Bad {central_type} {vel_disp_type}_velocity_dispersion_cm')
        #             # prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))
        #             append_to_dataset(group, f'{vel_disp_type}_velocity_dispersion_cm', np.nan, curr_length)
    
        #     for age_type in central_age_types:
        #         # first_time(prop_dict[central_type], f'{age_type}_stellar_age')
        #         create_dataset(group, f'{age_type}_stellar_age', dtype='f8', units='Gyr')
        #         try:
        #             # prop_dict[central_type][f'{age_type}_stellar_age'].append(central.ages[age_type].in_units('Gyr'))
        #             append_to_dataset(group, f'{age_type}_stellar_age', central.ages[age_type].in_units('Gyr'), curr_length)
        #         except:
        #             print(f'Bad {central_type} {age_type}_stellar_age')
        #             # prop_dict[central_type][f'{age_type}_stellar_age'].append(unyt.unyt_array(np.nan, 'Gyr'))
        #             append_to_dataset(group, f'{age_type}_stellar_age', np.nan, curr_length)
    
        #     for temp_type in central_temperature_types:
        #         # first_time(prop_dict[central_type], f'{temp_type}_temperature')
        #         create_dataset(group, f'{temp_type}_temperature', dtype='f8', units='K')
        #         try:
        #             # prop_dict[central_type][f'{temp_type}_temperature'].append(central.temperatures[temp_type].in_units('K'))
        #             append_to_dataset(group, f'{temp_type}_temperature', central.temperatures[temp_type].in_units('K'), curr_length)
        #         except:
        #             print(f'Bad {central_type} {temp_type}_temperature')
        #             # prop_dict[central_type][f'{temp_type}_temperature'].append(unyt.unyt_array(np.nan, 'K'))
        #             append_to_dataset(group, f'{temp_type}_temperature', np.nan, curr_length)

        #     print()


    
    
        ## dynamically deallocate variables to free up memory
        del target_snap, target_obj, halo, central
        gc.collect()
        print('\n\n\n\n')
    
    
    
        # Save properties dictionary
        # print('\nSaving properties dictionary to file')
        # time_start = timer()
        # save_object_with_dill(prop_dict, args.output_file, mode='wb+')
        # # dill.dump(prop_dict, f, dill.HIGHEST_PROTOCOL)
        # time_end = timer()
        # print(args.output_file)
        # print(f'Time to save file: {time_end-time_start} s')
        # print('DONE\n')


# print()
# pprint.pprint(prop_dict)
# print()


# ## A few extra properties that can be calculated from those already saved
# print('\nCalulating a few extra properties...\n')
# with h5py.File(args.output_file, 'r+') as f:
    
#     # for halo_type in halo_types:
#     #     try:
#     #         group = f[f'/{halo_type}']
#     #     except:
#     #         print(f'{halo_type} does not exist!')
#     #         continue
#     #     # create_dataset(group, 'ssfr', shape=(group['sfr'].shape[0],), dtype='f8')
#     #     # group['ssfr'].resize((group['sfr'].shape[0],))
#     #     # ssfr = unyt.unyt_array(group['sfr'])/unyt.unyt_array(group['stellar_mass'])
#     #     # group['ssfr'][:] = ssfr
    
#     halo_exists = True
#     try:
#         group = f[f'/halo']
#     except:
#         print(f'halo does not exist!')
#         halo_exists = False
    
#     if halo_exists:
#         try:
#             create_dataset(group, 'ssfr', shape=(group['sfr'].shape[0],), dtype='f8', units='yr**-1')
#             group['ssfr'].resize((group['sfr'].shape[0],))
#             ssfr = unyt.unyt_array(group['sfr'])/unyt.unyt_array(group['stellar_mass'])
#             group['ssfr'][:] = ssfr
#             # prop_dict[halo_type]['ssfr'] = unyt.unyt_array(prop_dict[halo_type]['sfr'])/unyt.unyt_array(prop_dict[halo_type]['stellar_mass'])

#             create_dataset(group, 'ssfr_100', shape=(group['sfr_100'].shape[0],), dtype='f8', units='yr**-1')
#             group['ssfr_100'].resize((group['sfr_100'].shape[0],))
#             ssfr_100 = unyt.unyt_array(group['sfr_100'])/unyt.unyt_array(group['stellar_mass'])
#             group['ssfr_100'][:] = ssfr_100
#             # prop_dict[halo_type]['ssfr_100'] = unyt.unyt_array(prop_dict[halo_type]['sfr_100'])/unyt.unyt_array(prop_dict[halo_type]['stellar_mass'])
#         except Exception as error:
#             print(f'Error calculating extra halo props: {error}')
#             # continue
    
#     # for central_type in central_types:
#     #     try:
#     #         group = f[f'/{central_type}']
#     #     except:
#     #         print(f'{central_type} does not exist!')
#     #         continue

#     central_exists = True
#     try:
#         group = f[f'/central']
#     except:
#         print(f'central does not exist!')
#         central_exists = False
    
#     if central_exists:
#         try:
#             create_dataset(group, 'ssfr', shape=(group['sfr'].shape[0],), dtype='f8', units='yr**-1')
#             group['ssfr'].resize((group['sfr'].shape[0],))
#             ssfr = unyt.unyt_array(group['sfr'])/unyt.unyt_array(group['stellar_mass'])
#             group['ssfr'][:] = ssfr
#             # prop_dict[central_type]['ssfr'] = unyt.unyt_array(prop_dict[central_type]['sfr'])/unyt.unyt_array(prop_dict[central_type]['stellar_mass'])

#             create_dataset(group, 'ssfr_100', shape=(group['sfr_100'].shape[0],), dtype='f8', units='yr**-1')
#             group['ssfr_100'].resize((group['sfr_100'].shape[0],))
#             ssfr_100 = unyt.unyt_array(group['sfr_100'])/unyt.unyt_array(group['stellar_mass'])
#             group['ssfr_100'][:] = ssfr_100
#             # prop_dict[central_type]['ssfr_100'] = unyt.unyt_array(prop_dict[central_type]['sfr_100'])/unyt.unyt_array(prop_dict[central_type]['stellar_mass'])

#             create_dataset(group, 'bh_mdot_edd', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='Msun/yr')
#             group['bh_mdot_edd'].resize((group['bh_mdot'].shape[0],))
#             bh_mdot_edd = unyt.unyt_array(group['bh_mdot'])/unyt.unyt_array(group['bh_fedd'])
#             group['bh_mdot_edd'][:] = bh_mdot_edd
#             # prop_dict[central_type]['bh_mdot_edd'] = unyt.unyt_array(prop_dict[central_type]['bh_mdot'])/unyt.unyt_array(prop_dict[central_type]['bh_fedd'])

#             create_dataset(group, f'bh_Lbol', shape=(group[f'bh_mdot'].shape[0],), dtype='f8', units='erg/s')
#             group[f'bh_Lbol'].resize((group[f'bh_mdot'].shape[0],))
#             # eta = 0.1
#             Lbol = (f_rad * unyt.c**2 *  unyt.unyt_array(group['bh_mdot'])).in_units('erg/s')
#             group[f'bh_Lbol'][:] = Lbol

#             create_dataset(group, 'bh_mdot_acc', shape=(group['bh_mdot'].shape[0],), dtype='f8', units='erg/s')
#             group['bh_mdot_acc'].resize((group[f'bh_mdot'].shape[0],))
#             bh_mdot_acc = unyt.unyt_array(group['bh_mdot']) / (1 - f_rad)
#             group['bh_mdot_acc'][:] = bh_mdot_acc

#             create_dataset(group, f'bh_Lbol_acc', shape=(group[f'bh_mdot_acc'].shape[0],), dtype='f8', units='erg/s')
#             group[f'bh_Lbol_acc'].resize((group[f'bh_mdot_acc'].shape[0],))
#             Lbol = (f_rad * unyt.c**2 *  unyt.unyt_array(group['bh_mdot_acc'])).in_units('erg/s')
#             group[f'bh_Lbol_acc'][:] = Lbol
#         except Exception as error:
#             print(f'Error calculating extra central props: {error}')
#             # continue
        
    


## Save properties dictionary
# print('\nSaving properties dictionary to file')
# save_object_with_dill(prop_dict, args.output_file, mode='wb+')
# print(args.output_file)
# print('DONE\n')