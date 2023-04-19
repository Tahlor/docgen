import h5py
# Create an HDF5 file
with h5py.File('data.h5', 'r') as hf:
    # List all groups
    print("Keys: %s" % hf.keys())
    a_group_keys = list(hf.keys())
    for a_group_key in a_group_keys:
        # Get the data
        print(hf[a_group_key][0:4])


