import jittor as jt
from jittor import init
from jittor import nn, dataset
import glob
import os
import numpy as np
import scipy.io as sio

class SPVAEDataset(dataset.Dataset):
    def __init__(self, mat_dir, mode):
        super(SPVAEDataset, self).__init__()
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, (self.mode + '.lst')), dtype=str)
        self.files = []
        for folder in self.folders:
            self.files = (self.files + glob.glob(os.path.join(self.mat_dir, folder, 'geo_zs.mat')))
        self.files = [file for file in self.files if ('acap' not in file)]
        self.files = sorted(self.files)
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fullname = self.files[idx]
        geo_data = sio.loadmat(fullname, verify_compressed_data_integrity=False)
        geo_zs = geo_data['geo_zs']
        return (geo_zs, fullname)
