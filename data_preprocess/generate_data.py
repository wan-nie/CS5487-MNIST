import os
import numpy as np
import pandas as pd



if __name__ == '__main__':
    raw_data_dir = '../data/raw_data/'
    processed_data_dir = '../data/processed_data'

    #
    data_path = os.path.join(raw_data_dir, 'digits4000_txt/digits4000_digits_vec.txt')
    label_path = os.path.join(raw_data_dir, 'digits4000_txt/digits4000_digits_labels.txt')
    cdata_path = os.path.join(raw_data_dir, 'challenge/cdigits_digits_vec.txt')
    clabel_path = os.path.join(raw_data_dir, 'challenge/cdigits_digits_labels.txt')

    # data4000
    vectors = pd.read_csv(data_path, sep='\t', header=None).values
    vectors = vectors.reshape((-1, 28, 28)).transpose(0, 2, 1)
    labels = pd.read_csv(label_path, sep='\t', header=None).values.reshape(-1)
    np.save(os.path.join(processed_data_dir, 'vectors.npy'), vectors)
    np.save(os.path.join(processed_data_dir, 'labels.npy'), labels)

    # cdata
    vectors = pd.read_csv(cdata_path, sep='\t', header=None).values
    vectors = vectors.reshape((-1, 28, 28)).transpose(0, 2, 1)
    labels = pd.read_csv(clabel_path, sep='\t', header=None).values.reshape(-1)
    np.save(os.path.join(processed_data_dir, 'cvectors.npy'), vectors)
    np.save(os.path.join(processed_data_dir, 'clabels.npy'), labels)

