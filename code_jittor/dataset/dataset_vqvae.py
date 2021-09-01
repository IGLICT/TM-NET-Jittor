import glob
import os

import jittor as jt
import numpy as np
from Augmentor.Operations import Distort
from jittor import dataset, init, nn
from PIL import Image


class VQVAEDataset(dataset.Dataset):
    def __init__(self, image_dir, mode, category=None, part_name=None, height=256, width=256):
        super(VQVAEDataset, self).__init__()
        self.image_dir = image_dir
        self.mode = mode
        self.category = category
        self.part_name = part_name
        self.height = int(height)
        self.width = int(width)
        if (self.part_name is None):
            self.part_name = ''
        if ((self.mode == 'train') or (self.mode == 'val') or (self.mode == 'test')):
            self.distort_aug = Distort(probability=1, grid_height=3, grid_width=4, magnitude=0)
        self.transform = self.get_transform(self.mode, (self.height * 3), (self.width * 4), self.category)
        self.folders = np.loadtxt(os.path.join(self.image_dir, (self.mode + '.lst')), dtype=str)
        self.files = []
        for folder in self.folders:
            no_patch_files = list((set(glob.glob(os.path.join(self.image_dir, folder, '*.png'))) - set(glob.glob(os.path.join(self.image_dir, folder, '*patch*.png')))))
            no_patch_files = [filename for filename in no_patch_files if (self.part_name in filename)]
            self.files = (self.files + no_patch_files)
        self.files = sorted(self.files)
        print('model num: {}'.format(len(self.files)))
        self.H_begin = [0, 256, 256, 256, 256, 512]
        self.W_begin = [256, 0, 256, 512, 768, 256]

    def __len__(self):
        return len(self.files)

    def get_transform(self, mode, height, width, category):
        if (category == 'car'):
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.5, 0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5, 0.5]
        if ((mode == 'train') or 'val'):
            transform = jt.transform.Compose([
                jt.transform.Resize((height, width)), 
                jt.transform.CenterCrop((height, width)), 
                jt.transform.ToTensor(), 
                jt.transform.ImageNormalize(mean, std)
                ])
        elif (mode == 'test'):
            transform = jt.transform.Compose([
                jt.transform.Resize((height, width)), 
                jt.transform.CenterCrop((height, width)), 
                jt.transform.ToTensor(), 
                jt.transform.ImageNormalize(mean, std)
                ])
        return transform

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        filename = self.files[idx]
        basename = os.path.basename(filename)
        image = Image.open(filename)
        if ((self.mode == 'train') or (self.mode == 'val')):
            np_image = np.array(image)
            distorted_image = np.zeros((image.size[1], image.size[0], np_image.shape[2]))
            for i in range(6):
                patch = Image.fromarray(np.uint8(np_image[self.H_begin[i]:(self.H_begin[i] + 256), self.W_begin[i]:(self.W_begin[i] + 256), :]))
                distorted_patch = self.distort_aug.perform_operation([patch])
                distorted_image[self.H_begin[i]:(self.H_begin[i] + 256), self.W_begin[i]:(self.W_begin[i] + 256), :] = np.array(distorted_patch[0])
            image = Image.fromarray(np.uint8(distorted_image))
        np_image = np.array(image)
        np_image.setflags(write=1)
        if (self.category == 'car'):
            image = Image.fromarray(np.uint8(np_image[:, :, 0:3]))
        else:
            image = Image.fromarray(np.uint8(np_image[:, :, 0:4]))
        if (self.transform is not None):
            image = self.transform(image)
        if (self.category == 'car'):
            patches = jt.zeros((6, 3, self.height, self.width))
        else:
            patches = jt.zeros((6, 4, self.height, self.width))
        basenames = []
        for i in range(6):
            patches[i, :, :, :] = image[:, self.H_begin[i]:(self.H_begin[i] + 256), self.W_begin[i]:(self.W_begin[i] + 256)]
            basenames.append('{}_patch{}.png'.format(basename.split('.')[0], str(i)))
        return (patches, basenames)

if __name__ == '__main__':
    data_root = '/mnt/f/wutong/data/car_new_reg'
    category = 'car'
    part_name = 'body'
    height = 256
    width = 256
    parallel = 'False'
    mode = 'val'
    batch_size = 6

    dataset = VQVAEDataset(
                    data_root, 
                    mode, 
                    category=category, 
                    part_name=part_name, 
                    height=height, 
                    width=width)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader = dataset.set_attrs(batch_size=batch_size, shuffle=False, drop_last=True)
    for i, (img, filename) in enumerate(dataloader):
        print(img.shape)
        # from einops import rearrange, reduce, repeat
        # img = rearrange(img, 'B P C H W -> (B P) C H W')
        shape = img.shape
        img = img.reshape((shape[0]*shape[1], shape[2], shape[3], shape[4]))
        print(img.shape)
