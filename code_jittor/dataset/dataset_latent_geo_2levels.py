import jittor as jt
from jittor import init
from jittor import nn
from jittor import dataset
import glob
import os
import numpy as np
import scipy.io as sio

class LatentGeo2LevelsDataset(dataset.Dataset):

    def __init__(self, mat_dir, mode, part_name=None):
        super(LatentGeo2LevelsDataset, self).__init__()
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, (self.mode + '.lst')), dtype=str)
        self.part_name = part_name
        if (self.part_name is None):
            self.part_name = ''
        self.files = []
        for folder in self.folders:
            self.files = (self.files + glob.glob(os.path.join(self.mat_dir, folder, (('*' + self.part_name) + '.mat'))))
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
        return (geo_z, id_ts, id_bs, fullname)

if (__name__ == '__main__'):
    data_root = './20210422_table_latents'
    category = 'table'
    part_name = 'surface'
    height = 256
    width = 256
    parallel = 'False'
    mode = 'train'
    batch_size = 6
    dataset = LatentGeo2LevelsDataset(data_root, mode, part_name=part_name)
    dataloader = dataset.set_attrs(batch_size=batch_size, shuffle=False, drop_last=True)
    for (i, (geo_z, id_ts, id_bs, fullname)) in enumerate(dataloader):
        print('{} {} {} {}'.format(fullname, geo_z.shape, id_ts.shape, id_bs.shape))