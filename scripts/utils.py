import numpy as np
import unyt


def parse_pos(pos_str):
    """Parse position string like '(186373.562,158985.797,172568.719)' into tuple of floats."""
    pos_str = pos_str.decode('utf-8') if isinstance(pos_str, bytes) else pos_str
    pos_str = pos_str.strip().strip('()')
    return tuple(map(float, pos_str.split(',')))


def identify_halo_from_file(ahf_halo_id):
    """Identify the halo from the AHF halo ID.

    Args:
        ahf_halo_id (int): The AHF halo ID.
    Returns:
        tuple: A tuple containing the following halo properties:
            - halo_m200c_z0 (unyt.unyt_array): Halo mass M200c at z=0 in Msun.
            - halo_zform (float): Halo formation redshift.
            - halo_R200m_z0 (unyt.unyt_array): Halo radius R200m at z=0 in kpc.
            - halo_pos (unyt.unyt_array): Halo position in kpc.
            - halo_dist_to_nearest_neighbor (unyt.unyt_array): Distance to nearest neighbor in kpc.
    """

    halo_file = '/scratch/aspadawe/snapshots/Hyenas/weiguang_halo_info/select_halo_ids.txt'
    # halo_info = np.genfromtxt(halo_file, comments='#', skip_header=4, names=True, replace_space='-')#, names=['halo_id', 'halo_name'])
    halo_info = np.genfromtxt(halo_file, comments='#', #replace_space='-',
                              dtype=[('log10z_0-M200c', 'f8'), ('log10z_0.5', 'f8'), ('GroupID', 'i8'), ('R_Mean200', 'f8'), ('Pos', 'O'), ('Dist.-to-nearest-neighbor', 'f8')],
                              converters={4: parse_pos}
                            #   names=['log10(z_0 M200c)', 'log10(z_0.5) (formation based on M200c)', 'GroupID', 'R_Mean200', 'Pos', 'Dist. to nearest neighbor']
                              )
    # with open(halo_file, 'r') as f:
    print()
    print(halo_info.dtype)
    print(halo_info)
    print()

    ahf_halo_ids = halo_info['GroupID']
    # ahf_halo_ids = halo_info[:,2]
    print('ahf_halo_ids:', ahf_halo_ids, '\n')

    # halo_m200c_z0 = unyt.unyt_array(10**halo_info['log10(z_0-M200c)'], 'Msun')
    # halo_zform = 10**halo_info['log10(z_0.5)-(formation-based-on-M200c)']
    # halo_R200m_z0 = unyt.unyt_array(halo_info['R_Mean200'], 'kpc')
    # halo_pos = unyt.unyt_array(halo_info['Pos'], 'kpc')
    # halo_dist_to_nearest_neighbor = unyt.unyt_array(halo_info['Dist.-to-nearest-neighbor'], 'kpc')

    halo_index = np.nonzero(ahf_halo_ids == ahf_halo_id)[0][0]
    print(f'Identified ahf halo index: {halo_index}\n')

    halo_m200c_z0 = unyt.unyt_array(10**halo_info['log10z_0M200c'], 'Msun')[halo_index]
    halo_zform = 10**halo_info['log10z_05'][halo_index]
    halo_R200m_z0 = unyt.unyt_array(halo_info['R_Mean200'], 'kpc')[halo_index]
    halo_pos = unyt.unyt_array(halo_info['Pos'][halo_index], 'kpc')
    print('halo_pos:', halo_pos, '\n')
    halo_dist_to_nearest_neighbor = unyt.unyt_array(halo_info['Disttonearestneighbor'], 'kpc')[halo_index]
    
    # halo_m200c_z0 = unyt.unyt_array(10**halo_info[:,0], 'Msun')[halo_index]
    # halo_zform = 10**halo_info[:,1][halo_index]
    # halo_R200m_z0 = unyt.unyt_array(halo_info[:,3], 'kpc')[halo_index]
    # halo_pos = unyt.unyt_array(halo_info[:,4], 'kpc')[halo_index]
    # halo_dist_to_nearest_neighbor = unyt.unyt_array(halo_info[:,5], 'kpc')[halo_index]
    
    return (halo_m200c_z0, halo_zform, halo_R200m_z0, halo_pos, halo_dist_to_nearest_neighbor)


def identify_L1_halo_from_file(ahf_halo_id):
    """Identify the halo from the AHF halo ID.

    Args:
        ahf_halo_id (int): The AHF halo ID.
    Returns:
        int: The corresponding Caesar halo ID.
    """

    halo_file = '/scratch/aspadawe/snapshots/Hyenas/weiguang_halo_info/Matched_Elite_halos-at-z0-Level1_M200c.txt'
    halo_info = np.genfromtxt(halo_file, comments='#')#, skip_header=4, names=True, replace_space='-')#, names=['halo_id', 'halo_name'])
    # with open(halo_file, 'r') as f:
    print()
    # print(halo_info.dtype.names)
    print(halo_info)
    print()

    ahf_halo_ids = halo_info[:,0]
    print('ahf_halo_ids:', ahf_halo_ids, '\n')

    halo_index = np.nonzero(ahf_halo_ids == ahf_halo_id)[0][0]
    print(f'Identified ahf halo index: {halo_index}\n')

    caesar_halo_id = int(halo_info[:,1][halo_index])
    print(f'Corresponding Caesar halo ID: {caesar_halo_id}\n')

    return caesar_halo_id