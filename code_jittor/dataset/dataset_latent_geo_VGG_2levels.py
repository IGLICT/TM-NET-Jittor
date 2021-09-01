import jittor as jt
from jittor import init
from jittor import nn
from jittor import dataset
import glob
import os
from collections import namedtuple
import numpy as np
import scipy.io as sio

class LatentGeoVGG2LevelsDataset(dataset.Dataset):

    def __init__(self, mat_dir, mode, part_name=None):
        super(LatentGeoVGG2LevelsDataset, self).__init__()
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, (self.mode + '.lst')), dtype=str)
        self.part_name = part_name
        if (self.part_name is None):
            self.part_name = ''
        self.files = []
        for folder in self.folders:
            self.files = (self.files + glob.glob(os.path.join(self.mat_dir, folder, (('*' + self.part_name) + '*.mat'))))
        self.files = [file for file in self.files if ('acap' not in file)]
        self.files = sorted(self.files)
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fullname = self.files[idx]
        data_dict = sio.loadmat(fullname, verify_compressed_data_integrity=False)
        geo_z = data_dict['geo_z']
        id_ts = data_dict['id_ts']
        id_bs = data_dict['id_bs']
        central_vggs = data_dict['central_vggs']
        return (geo_z, id_ts, id_bs, central_vggs, fullname)