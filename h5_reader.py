import h5py

file_dir = './models/prometheus/model_weights.h5'

def read_model_weights(weight_file_path):
    '''
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze

    Author(s): Moto Mthrok, Mjshi Jewes
    '''

    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print('{} contains: '.format(weight_file_path))
            print('Root attributes:')
        for key, value in f.attrs.items():
            print('  {}: {}'.format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print('  {}'.format(layer))
            print('    Attributes:')
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print('    Dataset:')
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print('      {}/{}: {}'.format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()

if __name__ == '__main__':
    print('Reading file...')
    read_model_weights(read_model_weights)