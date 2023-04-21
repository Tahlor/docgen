import h5py

def A():
    # create sample data for group1
    group1_data = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }

    # create sample data for group2
    group2_data = {
        'd': [10, 11, 12],
        'e': [13, 14, 15]
    }

    # write sample data to an HDF5 file
    with h5py.File('test.h5', 'w') as file:
        # create group1 and write data to it
        group1 = file.create_group('group1')
        for dataset_name, data in group1_data.items():
            group1.create_dataset(dataset_name, data=data)

        # create group2 and write data to it
        group2 = file.create_group('group2')
        for dataset_name, data in group2_data.items():
            group2.create_dataset(dataset_name, data=data)

    # open the HDF5 file in read-write mode
    with h5py.File('test.h5', 'r+') as file:
        # get a reference to group1 and group2
        group1 = file['group1']
        group2 = file['group2']

        # # move each dataset in group2 to group1
        file['group1'].update(file['group2'])
        del file['group2']

        # print the contents of group1
        print("Contents of group1:")
        for dataset_name in group1:
            print(dataset_name)

        # write a new dataset to group1
        group1.create_dataset('new_dataset', data=[16, 17, 18])

        # print the contents of group1 again
        print("Contents of group1 after adding new_dataset:")
        for dataset_name in group1:
            print(dataset_name)


import numpy as np
import h5py

filename = rf"G:\synthetic_data\one_line\french\000000000.jpg"
fin = open(filename, 'rb')
binary_data = fin.read()

with h5py.File('foo.hdf5', "w") as f:
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('binary_data', (100, ), dtype=dt)
    # Save data string converted as a np array
    dset[0] = np.fromstring(binary_data, dtype='uint8')