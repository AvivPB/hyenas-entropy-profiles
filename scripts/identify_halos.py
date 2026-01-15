## For dougpython: module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

import yt
import unyt

import caesar
import numpy as np
# import matplotlib.pyplot as plt

import os
from pathlib import Path
import argparse

import utils


parser = argparse.ArgumentParser(prog='identify_halos.py', description='Find halo id and minpotpos of target halo in simulation based on provided halo properties and contamination (optional) from caesar file')

# parser.add_argument('--snap_dir', action='store', type=str, required=True, 
#                     help='directory containing snapshots')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory containing caesar files')
parser.add_argument('--snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers')
parser.add_argument('--caesar_base', action='store', type=str, default='caesar_',
                    help='Base name for caesar files, e.g. caesar_')
parser.add_argument('--caesar_suffix', action='store', type=str, default='',
                    help='Suffix for caesar files, e.g. _haloid-fof_lowres-[2]')

parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
# parser.add_argument('--clear_output_file', action='store', type=bool, default=False, choices=[True, False],
                    # help='Whether to clear the output file initially before writing to it')
parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
                    help='Whether to clear the output file initially before writing to it')

parser.add_argument('--target_property', action='store', nargs='*', type=str, default=[],
                    choices=['m2500c', 'm500c', 'm200c', 'r2500c', 'r500c', 'r200c', 'temperature'],
                    help='Halo property employed to identify target halo')
parser.add_argument('--domain', action='store', type=str, nargs='*', default=[],
                     choices=['inside', 'outside'],
                    help='Look for halos with target property inside or outside min and max target values')
parser.add_argument('--target_value_min', action='store', nargs='*', type=float, default=[],
                    help='Minimum value of target_property')
parser.add_argument('--target_value_max', action='store', nargs='*', type=float, default=[],
                    help='Maximum value of target_property')
parser.add_argument('--target_units', action='store', nargs='*', type=str, default=[],
                    help='Units of target_property')
# parser.add_argument('--target_value', action='store',
#                     help='Value of target_property')
# parser.add_argument('--target_value_tol', action='store', type=float,
#                     help='Tolerance of target halo value as a fraction, i.e. all halos with target_property value within +/- target_value_tol*target_value (inclusive) will be included')
# parser.add_argument('--target_value_is_log', action='store', type=bool, choices=[True, False],
#                     help='Whether provided value of target_property is linear (False) or log (True)')

# parser.add_argument('--use_contamination', action='store', type=bool, choices=[True, False],
#                     help='Use lowres contamination property to select halo')
parser.add_argument('--use_contamination', action=argparse.BooleanOptionalAction, default=False,
                    help='Use lowres contamination property to select halo')
parser.add_argument('--return_contamination', action=argparse.BooleanOptionalAction, default=True,
                    help='Return lowres contamination property')
parser.add_argument('--contamination_min', action='store', type=float, default=0,
                    help='Minimum contamination of target halo')
parser.add_argument('--contamination_max', action='store', type=float, default=0,
                    help='Maximum contamination of target halo')
# parser.add_argument('--contamination', action='store', type=float,
#                     help='Contamination of target halo')
# parser.add_argument('--contamination_tol', action='store', type=float,
                    # help='Tolerance of target halo contamination as a fraction, i.e. all halos with contamination within +/- contamination_tol*contamination (inclusive) will be included')

parser.add_argument('--use_dist', action=argparse.BooleanOptionalAction, default=False,
                    help='Use minpotpos of halos to get distances to centre of gas particles')
parser.add_argument('--return_dist', action=argparse.BooleanOptionalAction, default=True,
                    help='Return halo distances to centre of gas particles')
parser.add_argument('--pos_units', action='store', type=str, default='Mpc',
                    help='Units to use for position')
parser.add_argument('--snap_dir', action='store', type=str, required=False, 
                    help='directory containing snapshot files (only needed if use_dist)')
parser.add_argument('--snap_base', action='store', type=str, default='snapshot_',
                    help='Base name for snapshot files, e.g. snapshot_')

parser.add_argument('--return_pos', action=argparse.BooleanOptionalAction, default=True, 
                    help='Return minpotpos of identified halos')

parser.add_argument('--use_halo_files', action=argparse.BooleanOptionalAction, default=False,
                    help='Use external halo files to identify halos')
parser.add_argument('--try_L1_halo_file', action=argparse.BooleanOptionalAction, default=False,
                    help='Try to identify halos from Level 1 halo file')
parser.add_argument('--ahf_halo_id', action='store', type=int, required=False, default=None,
                    help='AHF halo ID to identify target halo from external halo files')
parser.add_argument('--tolerance', action='store', type=float, required=False, default=0.1,
                    help='Tolerance when matching halo properties from external halo files (as a fraction, e.g. 0.1 for 10 percent)')
parser.add_argument('--distance_tolerance', action='store', type=float, required=False, default=0.5,
                    help='Distance tolerance when matching halo positions from external halo files (in Mpc)')


args = parser.parse_args()

# print()
# print(args.clear_output_file)
# print(args.use_contamination)
# print()


# if args.target_value_is_log:
#     target_units = None
# else:
#     target_units = args.target_units

# target_value = unyt.unyt_array(args.target_value, target_units)
# target_value_min = target_value * (1 - unyt.unyt_array(args.target_value_tol, target_units))
# target_value_max = target_value * (1 + unyt.unyt_array(args.target_value_tol, target_units))

# contamination_min = args.contamination * (1 - args.contamination_tol)
# contamination_max = args.contamination * (1 + args.contamination_tol)




def gas_centre(snap_data):
    gas_pos = snap_data['gas', 'position'].in_units(args.pos_units)
    gas_centre = np.average(gas_pos, axis=0)
    print(gas_centre)
    assert np.shape(gas_centre)[-1] == 3, "Incorrect shape for gas_centre"
    return gas_centre




if not os.path.exists(args.output_file):
    print('Making output file')
    # output_file = Path(args.output_file)
    # output_file.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    f = open(args.output_file, 'w')
    f.close()
    print()

if args.clear_output_file:
    print('Clearing output file')
    f = open(args.output_file, 'w')
    f.close()
    print()

if not os.path.exists(f'{args.output_file}.header'):
    print('Making output file header')
    # os.makedirs(f'{args.output_file}.header')
    f = open(f'{args.output_file}.header', 'w')
    f.close()
    print()

print('Writing to output file header')
# header = f'caesar_dir\tsnap_num\tz\thalo_id\tminpotpos_x (Mpccm/h)\tminpotpos_y (Mpccm/h)\tminpotpos_z (Mpccm/h)'
header = f'caesar_dir\tsnap_num\tz\thalo_id'
for target_property, target_units in zip(args.target_property, args.target_units):
    header += f'\t{target_property} ({target_units})'
if args.return_contamination:
    header += '\tcontamination'
if args.return_dist:
    header += f'\tDistance ({args.pos_units})'
if args.return_pos:
    header += f'\tminpotpos_x ({args.pos_units})\tminpotpos_y ({args.pos_units})\tminpotpos_z ({args.pos_units})'
with open(f'{args.output_file}.header', 'w') as f:
    f.write(header)
print()


# target_value_min = unyt.unyt_array(args.target_value_min, args.target_units)
# target_value_max = unyt.unyt_array(args.target_value_max, args.target_units)

# print()
# print(target_value_min)
# print(target_value_max)
# print()

for snap_num in args.snap_nums:
    # snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{snap_num:03}.hdf5')
    # snap = yt.load(snap_file)

    ## Load snapshot file
    if args.use_dist or args.return_dist:
        snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{snap_num:03}.hdf5')
        try:
            snap = yt.load(snap_file)
        except Exception as e:
            print(f'Error reading snap file {snap_file}: {e}')
            continue
        snap_data = snap.all_data()
        gas_centre = gas_centre(snap_data)

    ## Load caesar file
    caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{snap_num:03}{args.caesar_suffix}.hdf5')
    try:
        obj = caesar.load(caesar_file)
    except Exception as e:
        print(f'Error reading caesar file {caesar_file}: {e}')
        continue

    
    ## Get all relevant halo properties ##############

    # halo_ids = unyt.unyt_array([halo.GroupID for halo in obj.halos])
    halo_ids = np.array([halo.GroupID for halo in obj.halos])
    
    # halo_minpotposes = unyt.unyt_array([halo.minpotpos.in_units('Mpccm/h') for halo in obj.halos])
    # halo_minpotposes = np.array([halo.minpotpos.in_units('Mpccm/h') for halo in obj.halos])   ## Good for printing out

    halo_prop_dict = {
        target_property:np.array([halo.virial_quantities[target_property].in_units(target_units) for halo in obj.halos]) for target_property, target_units in zip(args.target_property, args.target_units)
    }
    # print()
    # print(halo_prop_dict)
    # print()
    # halo_prop_values = unyt.unyt_array([halo.virial_quantities[args.target_property].in_units(args.target_units) for halo in obj.halos])
    # halo_prop_values = np.array([halo.virial_quantities[args.target_property].in_units(args.target_units) for halo in obj.halos])
    # print(halo_prop_values)
    # print(halo_prop_values.units)
    # print()
    # if args.target_value_is_log:
    #     halo_prop_value = np.log10(halo_prop_value)
        
    # halo_contaminations = unyt.unyt_array([halo.contamination for halo in obj.halos])
    if args.use_contamination or args.return_contamination:
        halo_contaminations = np.array([halo.contamination for halo in obj.halos])


    if args.use_dist or args.return_dist:
        halo_distances = np.array([np.linalg.norm(halo.minpotpos.in_units(args.pos_units) - gas_centre.in_units(args.pos_units)) for halo in obj.halos])

    if args.return_pos:
        halo_minpotposes = np.array([halo.minpotpos.in_units(args.pos_units) for halo in obj.halos])   ## Good for printing out

    


    ## Find desired halo ###########################

    # target_filter = (halo_prop_values >= target_value_min) & (halo_prop_values <= target_value_max) & (halo_contaminations >= contamination_min) & (halo_contaminations <= contamination_max)
    # target_filter = (halo_prop_values >= args.target_value_min) & (halo_prop_values <= args.target_value_max) & (halo_contaminations >= args.contamination_min) & (halo_contaminations <= args.contamination_max)
    target_filter = np.full(shape=len(obj.halos), fill_value=True)
    # print()
    # print(len(obj.halos))
    # print(target_filter)
    # print()



    ## Identify halos from external halo files
    if args.use_halo_files:
        if args.ahf_halo_id is None:
            raise ValueError('AHF halo ID must be provided when using external halo files to identify halos')
        print(f'\nIdentifying halo from external halo files using AHF halo ID {args.ahf_halo_id}\n')
        print(type(args.ahf_halo_id))
        print()
        

        obtained_caesar_halo_id = False
        if args.try_L1_halo_file:

            # caesar_halo_id = utils.identify_L1_halo_from_file(args.ahf_halo_id)
            # print(f'Identified Caesar halo ID from AHF halo ID {args.ahf_halo_id}: {caesar_halo_id}\n')
            # obtained_caesar_halo_id = True

            try:
                caesar_halo_id = utils.identify_L1_halo_from_file(args.ahf_halo_id)
                print(f'Identified Caesar halo ID from AHF halo ID {args.ahf_halo_id}: {caesar_halo_id}\n')
                obtained_caesar_halo_id = True
            except Exception as e:
                print(f'Error identifying Caesar halo ID from AHF halo ID {args.ahf_halo_id} using Level 1 halo file: {e}')
                # print('Trying standard halo file instead...\n')
                # try:
                #     caesar_halo_id = utils.identify_halo_from_file(args.ahf_halo_id)
                #     print(f'Identified Caesar halo ID from AHF halo ID {args.ahf_halo_id}: {caesar_halo_id}\n')
                # except Exception as e:
                #     print(f'Error identifying Caesar halo ID from AHF halo ID {args.ahf_halo_id} using standard halo file: {e}')
                #     continue
        
        if obtained_caesar_halo_id:
            target_filter = target_filter & (halo_ids == caesar_halo_id)
        else:
            halo_m200c_z0, halo_zform, halo_R200m_z0, halo_pos, halo_dist_to_nearest_neighbor = utils.identify_halo_from_file(args.ahf_halo_id)
            
            # h = obj.simulation.hubble_constant
            # print('snap.hubble_constant:', snap.hubble_constant)
            # print('h:', h, '\n')
            halo_R200m_z0 /= obj.simulation.hubble_constant  # Convert from kpc/h to kpc
            halo_pos /= obj.simulation.hubble_constant      # Convert from kpc/h to kpc
            halo_dist_to_nearest_neighbor /= obj.simulation.hubble_constant  # Convert from kpc/h to kpc
            # print(halo_R200m_z0.in_units('kpc/h'))

            print(f'Identified halo properties from AHF halo ID {args.ahf_halo_id}:\n')
            print(f'  M200c at z=0: {halo_m200c_z0}')
            print(f'  Formation redshift: {halo_zform}')
            print(f'  R200m at z=0: {halo_R200m_z0}')
            print(f'  Position: {halo_pos}')
            print(f'  Distance to nearest neighbor: {halo_dist_to_nearest_neighbor}\n')

            halo_m200c_values = unyt.unyt_array([halo.virial_quantities['m200c'].in_units('Msun') for halo in obj.halos], 'Msun')
            # halo_distances_to_file_pos = unyt.unyt_array([np.linalg.norm(halo.minpotpos.in_units(args.pos_units) - halo_pos.in_units(args.pos_units)) for halo in obj.halos], args.pos_units)
            # print()
            # print(halo_distances_to_file_pos)
            # print(min(halo_distances_to_file_pos))
            # print()

            target_filter = target_filter & (np.abs(halo_m200c_values - halo_m200c_z0) / halo_m200c_z0 <= args.tolerance)
            print(target_filter)
            # target_filter = target_filter & (halo_distances_to_file_pos <= unyt.unyt_array(args.distance_tolerance, 'Mpc'))
            # print(target_filter)

            if args.use_contamination:
                target_filter = target_filter & (halo_contaminations >= args.contamination_min) & (halo_contaminations <= args.contamination_max)


        
        target_halo_indexes = np.where(target_filter)
        print(target_halo_indexes)
        target_halo_ids = halo_ids[target_halo_indexes]
        print(target_halo_ids)
        print()


        if not obtained_caesar_halo_id:
            ## Find halo with mass closest to that from file
            new_halo_m200c_values = halo_m200c_values[target_halo_indexes]
            new_target_filter = np.full(shape=len(new_halo_m200c_values), fill_value=True)
            # print()
            print(new_halo_m200c_values)
            # print(min(new_halo_m200c_values))
            # print(np.where(np.abs(new_halo_m200c_values - halo_m200c_z0) < 1e-10))
            print()
            # new_target_filter = new_target_filter & (np.abs(new_halo_m200c_values - halo_m200c_z0) < 1e-10)
            index_of_closest = np.argmin(np.abs(new_halo_m200c_values - halo_m200c_z0))
            print(f'Index of closest halo mass to that from file: {index_of_closest}')
            print(f'Closest halo mass to that from file: {new_halo_m200c_values[index_of_closest]}')
            print()
            new_target_filter = new_target_filter & (new_halo_m200c_values == new_halo_m200c_values[index_of_closest])
            # new_halo_indexes = np.array(np.where(target_filter)[0], dtype=int)
            new_halo_indexes = np.where(new_target_filter)#[0]
            print('new halo indexes:')
            print(new_halo_indexes)
            print()
            target_halo_indexes = target_halo_indexes[0][new_halo_indexes]
            print(target_halo_indexes)
            # target_halo_ids = target_halo_ids[target_halo_indexes]
            target_halo_ids = halo_ids[target_halo_indexes]
            print(target_halo_ids)
            print()




    ## Identify halos from provided halo properties
    else:
        for target_property, target_domain, target_value_min, target_value_max, in zip(args.target_property, args.domain, args.target_value_min, args.target_value_max):
            halo_prop_values = halo_prop_dict[target_property]
            if target_domain == 'inside':
                target_filter = target_filter & (halo_prop_values >= target_value_min) & (halo_prop_values <= target_value_max)
            elif target_domain == 'outside':
                target_filter = target_filter & (halo_prop_values <= target_value_min) & (halo_prop_values >= target_value_max)
        # target_filter = (halo_prop_values >= args.target_value_min) & (halo_prop_values <= args.target_value_max)
        
        if args.use_contamination:
            target_filter = target_filter & (halo_contaminations >= args.contamination_min) & (halo_contaminations <= args.contamination_max)
        
        
        
        # print()
        # print(np.where(target_filter))
        # print()
        target_halo_indexes = np.where(target_filter)#[0]
        print(target_halo_indexes)
        target_halo_ids = halo_ids[target_halo_indexes]



        # if args.use_dist or args.return_dist:
        #     halo_distances = np.array([np.linalg.norm(halo.minpotpos.in_units(args.pos_units) - gas_centre.in_units(args.pos_units)) for halo in obj.halos if halo.GroupID in target_halo_ids])

        if args.use_dist:
            new_halo_distances = halo_distances[target_halo_indexes]
            new_target_filter = np.full(shape=len(new_halo_distances), fill_value=True)
            print()
            print(new_halo_distances)
            print(min(new_halo_distances))
            print(np.where(new_halo_distances - min(new_halo_distances) < 1e-8))
            print()
            new_target_filter = new_target_filter & (new_halo_distances - min(new_halo_distances) < 1e-10)
            # new_halo_indexes = np.array(np.where(target_filter)[0], dtype=int)
            new_halo_indexes = np.where(new_target_filter)#[0]
            print('new halo indexes:')
            print(new_halo_indexes)
            target_halo_indexes = target_halo_indexes[0][new_halo_indexes]
            # target_halo_ids = target_halo_ids[target_halo_indexes]
            target_halo_ids = halo_ids[target_halo_indexes]

    


    ## Write to output file ###########################

    # target_halo_ids = halo_ids[target_halo_indexes]
    # target_halo_minpotposes = halo_minpotposes[target_halo_indexes]  ## Good for printing out
    # target_halo_prop_values = halo_prop_values[target_halo_indexes]
    target_halo_prop_dict = {
        # target_property:halo_prop_dict[target_property][target_filter] for target_property in args.target_property
        target_property:halo_prop_dict[target_property][target_halo_indexes] for target_property in args.target_property
    }
    if args.return_contamination:
        target_halo_contaminations = halo_contaminations[target_halo_indexes]
    if args.return_dist:
        target_halo_distances = halo_distances[target_halo_indexes]
    if args.return_pos:
        target_halo_minpotposes = halo_minpotposes[target_halo_indexes]   ## Good for printing out


    print(f'Writing to output file for snap {snap_num}')
    with open(args.output_file, 'a') as f:
        for index in range(len(target_halo_ids)):
            halo_id = target_halo_ids[index]
            # halo_minpotpos = target_halo_minpotposes[index]   ## Good for printing out
            halo_props = [target_halo_prop_dict[target_property][index] for target_property in args.target_property]
            
            # output = f'{args.caesar_dir}\t{snap_num}\t{obj.simulation.redshift}\t{halo_id}\t{halo_minpotpos[0]}\t{halo_minpotpos[1]}\t{halo_minpotpos[2]}'
            output = f'{args.caesar_dir}\t{snap_num}\t{obj.simulation.redshift}\t{halo_id}'
            for halo_prop in halo_props:
                output += f'\t{halo_prop}'
                
            if args.return_contamination:
                halo_contamination = target_halo_contaminations[index]
                output += f'\t{halo_contamination}'

            if args.return_dist:
                halo_distance = target_halo_distances[index]
                output += f'\t{halo_distance}'

            if args.return_pos:
                halo_minpotpos = target_halo_minpotposes[index]
                output += f'\t{halo_minpotpos[0]}\t{halo_minpotpos[1]}\t{halo_minpotpos[2]}'
                
            output += '\n'
            f.write(output)
            
        # for halo_id, halo_minpotpos, halo_contamination, halo_prop_value in zip(target_halo_ids, target_halo_minpotposes, target_halo_contaminations, target_halo_prop_values):
        #     f.write(f'{args.caesar_dir}\t{snap_num}\t{halo_id}\t{halo_minpotpos}\t{halo_contamination}\t{halo_prop_value}\n')

    print()
    print()

    

    

    # halo_nonzero_contamination_index = np.nonzero(np.array(halo_contamination)!=0)[0]
    # halo_zero_contamination_index = np.nonzero(np.array(halo_contamination)==0)[0]

    # halo_prop_value_nonzero_contamination = np.array(halo_prop_value)[halo_nonzero_contamination_index]
    # halo_prop_value_zero_contamination = np.array(halo_prop_value)[halo_zero_contamination_index]

    # target_halo_index = halo_contamination0_index[np.argmin(np.abs(halo_m500c_contamination0 - target_m500c))]