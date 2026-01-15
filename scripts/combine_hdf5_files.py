import h5py
import numpy as np
import glob
import re
import sys
from collections import defaultdict
from tqdm import tqdm  # Add progress bars

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_all_datasets(h5file):
    datasets = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj
    h5file.visititems(visitor)
    return datasets

def combine_hdf5_files(file_pattern, output_file, chunk_size=1000, enable_compression=False):
    # Get list of files and sort them
    files = glob.glob(file_pattern)
    files.sort(key=natural_sort_key)
    if not files:
        print("No files matched pattern:", file_pattern)
        return
    print(f"Found {len(files)} files to combine")
    
    # First pass: collect all dataset information
    all_datasets = set()
    dataset_info = {}
    total_lengths = defaultdict(int)
    
    print("Analyzing file structure...")
    for fname in tqdm(files, desc="Scanning files"):
        with h5py.File(fname, 'r') as f:
            datasets = get_all_datasets(f)
            all_datasets.update(datasets.keys())
            for dset_name, dset in datasets.items():
                if dset_name not in dataset_info:
                    dataset_info[dset_name] = {
                        'dtype': dset.dtype,
                        'shape': list(dset.shape),
                        'attrs': dict(dset.attrs)
                    }
                # assume concatenation along axis 0; guard if shape is empty
                if len(dset.shape) >= 1:
                    total_lengths[dset_name] += dset.shape[0]
                else:
                    # scalar dataset: count one per file (store as length +=1)
                    total_lengths[dset_name] += 1
    
    print(f"Found {len(all_datasets)} unique datasets")
    
    # Create output file and concatenate datasets
    print("Creating output file and copying data...")
    with h5py.File(output_file, 'w') as outf:
        # Create all groups first
        for dset_name in all_datasets:
            group_path = '/'.join(dset_name.split('/')[:-1])
            if group_path and group_path not in outf:
                outf.create_group(group_path)
        
        # Process datasets in sorted order for consistent progress
        for dset_name in sorted(all_datasets):
            info = dataset_info.get(dset_name)
            if info is None:
                print(f"Warning: no info for {dset_name}, skipping")
                continue
            orig_shape = info['shape']
            # Build combined shape: if original had ndim>=1, concat along axis 0
            if len(orig_shape) >= 1:
                combined_shape = list(orig_shape)
                combined_shape[0] = total_lengths[dset_name]
            else:
                # scalar datasets -> create 1D array of length = number of files that had it
                combined_shape = [total_lengths[dset_name]]
            
            # If total length zero, create empty dataset (no chunk/compression)
            if any(dim == 0 for dim in combined_shape):
                print(f"Dataset {dset_name} has zero dimension(s), creating empty dataset with shape {tuple(combined_shape)}")
                outf.create_dataset(dset_name, shape=tuple(combined_shape), dtype=info['dtype'])
                outds = outf[dset_name]
                for k, v in info['attrs'].items():
                    outds.attrs[k] = v
                continue
            
            print(f"\nProcessing dataset: {dset_name}  -> shape {tuple(combined_shape)}")
            
            # Decide whether to use chunking/compression: only if all dims > 0 and enable_compression True
            use_chunks = False
            chunks = None
            if all(d > 0 for d in combined_shape) and enable_compression:
                try:
                    # choose chunk dims: no larger than dataset dims
                    chunks = tuple(min(chunk_size, d) for d in combined_shape)
                    # ensure chunk dims are positive and <= dims
                    if any(c <= 0 or c > s for c, s in zip(chunks, combined_shape)):
                        chunks = None
                    else:
                        use_chunks = True
                except Exception:
                    chunks = None
                    use_chunks = False
            
            # Create dataset (no compression by default for speed)
            create_kwargs = {'dtype': info['dtype']}
            if use_chunks:
                create_kwargs['chunks'] = chunks
                create_kwargs['compression'] = 'gzip'
                create_kwargs['compression_opts'] = 4
            outds = outf.create_dataset(dset_name, shape=tuple(combined_shape), **create_kwargs)
            
            # Copy attributes
            for key, value in info['attrs'].items():
                outds.attrs[key] = value
            
            # Copy data file-by-file; use f.get to avoid visiting entire file
            pos = 0
            for fname in tqdm(files, desc=f"Copying {dset_name}", leave=False):
                try:
                    with h5py.File(fname, 'r') as f:
                        src = f.get(dset_name)
                        if src is None:
                            continue
                        # if original dataset had ndim>=1 we copy along axis 0
                        if len(orig_shape) >= 1:
                            n0 = src.shape[0]
                            if n0 == 0:
                                continue
                            outds[pos:pos + n0] = src[...]
                            pos += n0
                        else:
                            # scalar dataset: write one value at next position
                            val = src[()]
                            outds[pos] = val
                            pos += 1
                except Exception as e:
                    print(f"Error processing {fname} for dataset {dset_name}: {e}")
                    continue

    print(f"\nSuccessfully combined {len(files)} files into {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python combine_hdf5_files.py <file_pattern> <output_file>")
        sys.exit(1)
    
    combine_hdf5_files(sys.argv[1], sys.argv[2], chunk_size=1000, enable_compression=False)