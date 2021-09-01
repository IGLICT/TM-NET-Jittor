from jittor.utils.pytorch_converter import convert

pytorch_code="""
import argparse
import os
import pickle
from collections import namedtuple

import lmdb
import numpy as np
import scipy.io as sio
import torch
import torchvision.models as models
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import get_central_part_name, get_part_names
from dataset.dataset_geovae import GeoVAEDataset
from dataset.dataset_vqvae import VQVAEDataset
from networks.networks_geovae import GeoVAE
from networks.networks_vqvae import VQVAE
from config import load_config
from networks import get_network

def extract_latents_patch(loader, model, geo_model, vgg_model, args):
    index = 0
    H_begin = [0, 256, 256, 256, 256, 512]
    W_begin = [256, 0, 256, 512, 768, 256]
    if args.category == 'car':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.5, 0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize((args.height*3, args.width*4)),
            transforms.CenterCrop((args.height*3, args.width*4)),
            # data augmentation
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    args.vgg_height = 224
    args.vgg_width = 224
    vgg_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.CenterCrop((args.vgg_height, args.vgg_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pbar = tqdm(loader)

    for i, (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, filenames) in enumerate(pbar):
        geo_input = geo_input.to(args.device).float().contiguous()
        geo_zs, geo_outputs, _, _ = geo_model(geo_input)
        geo_zs = geo_zs.detach().cpu().numpy()
        
        for j in range(geo_input.shape[0]):
            filename = filenames[j]
            head_tail = os.path.split(filename)
            head = head_tail[0]
            basename = os.path.basename(filename)
            basename_without_ext = basename.split('.')[0]
            fid = basename_without_ext.split('_')[0]
            model_id = basename.split('_')[0]
            
            sub_dir = os.path.join(args.save_path, model_id)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            flag_exist = True
            # central image
            central_img_filename = os.path.join(head, fid+'_'+args.central_part_name+'.png')
            if os.path.exists(central_img_filename):
                central_image = Image.open(central_img_filename)
                np_central_image = np.array(central_image)
                np_central_image.setflags(write=1)
                np_central_image = np_central_image[:, :, 0:3]
                flag_exist = True
            else:
                print('warning: {} not exists'.format(central_img_filename))
                flag_exist = False

            # current part image
            img_filename = os.path.join(head, basename_without_ext+'.png')
            if os.path.exists(img_filename):
                ori_image = Image.open(img_filename)
                sub_dir = os.path.join(args.save_path, '{}'.format(model_id))
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                # distort
                np_image = np.array(ori_image)
                distorted_image = np.zeros((ori_image.size[1], ori_image.size[0], np_image.shape[2]))
                for i in range(6):
                    patch = Image.fromarray(np.uint8(np_image[H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256, :]))
                    # distorted_patch = distort_aug.perform_operation([patch])
                    # distorted_image[0].save(os.path.join('.', basename))
                    distorted_image[H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256, :] = np.array(patch)
                image = Image.fromarray(np.uint8(distorted_image))
                # transform
                np_image = np.array(image)
                np_image.setflags(write=1)
                if args.category == 'car':
                    image = Image.fromarray(np.uint8(np_image[:, :, 0:3]))
                else:
                    # np_image[:, :, 3] = np_image[:, :, 3]/255
                    # np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
                    # np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
                    # np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
                    # np_image[:, :, 3] = np_image[:, :, 3]*255
                    image = Image.fromarray(np.uint8(np_image[:, :, 0:4]))
            else:
                flag_exist = False

                
            if True == flag_exist:
                # central part
                central_patches = torch.zeros(6, 3, args.vgg_height, args.vgg_width)

                for i in range(6):
                    central_patches[i, :, :, :] = vgg_transform(
                        Image.fromarray(np.uint8(
                        np_central_image[H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256, :]))
                        )
                central_patches = central_patches.to(args.device)
                central_vggs = vgg_model(central_patches)
                central_vggs = central_vggs.detach().cpu().numpy()

                # current part
                image = transform(image)
                # save_image(image, os.path.join('.', basename))
                if args.category == 'car':
                    patches = torch.zeros(6, 3, args.height, args.width)
                else:
                    patches = torch.zeros(6, 4, args.height, args.width)

                for i in range(6):
                    patches[i, :, :, :] = image[:, H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256]
                # get indices
                patches = patches.to(args.device)
                quant_t, quant_b, diff, id_ts, id_bs = model.encode(patches)
                dec = model.decode_code(id_ts, id_bs)
                
                id_ts = id_ts.detach().cpu().numpy()
                id_bs = id_bs.detach().cpu().numpy()
                
                # save
                print('{} {} {} {}'.format(id_ts.shape, id_bs.shape, geo_zs[j:j+1, :].shape, central_vggs.shape))

                data_dict = {
                    'geo_z': geo_zs[j:j+1, :], 
                    'id_ts': id_ts,
                    'id_bs': id_bs,
                    'central_vggs': central_vggs
                }
                sio.savemat(os.path.join(sub_dir, basename), data_dict)
                # DEBUG save
                if dec.shape[1] == 4:
                    dec[:, 3, :, :] = ((dec[:, 3, :, :] > 0).float()-0.5)*2
                # from torchvision.utils import save_image, make_grid
                # save_image(make_grid(dec), os.path.join(sub_dir, '{}.png'.format( basename.split('.')[0])))
                # save_image(make_grid(patches), os.path.join(sub_dir, '{}_1.png'.format( basename.split('.')[0])))

                index += 1
                pbar.set_description(f'inserted: {basename}')
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)

    parser.add_argument('--vqvae_yaml', type=str, required=True)
    parser.add_argument('--vqvae_ckpt', type=str, required=True)

    parser.add_argument('--geovae_yaml', type=str, required=True)
    parser.add_argument('--geovae_ckpt_dir', type=str, required=True)

    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mat_dir', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    part_names = get_part_names(args.category)
    central_part_name = get_central_part_name(args.category)
    part_names.remove(central_part_name)
    args.central_part_name = central_part_name

    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    geovae_config = load_config(args.geovae_yaml)
    geovae_config['train']['device'] = args.device
    vqvae_config = load_config(args.vqvae_yaml)
    vqvae_config['train']['device'] = args.device
    # torch pretrained vgg model 
    vgg16 = models.vgg16(pretrained=True).to(args.device)
    vgg16.eval()

    for part_name in part_names:
        # geometry dataset
        geo_dataset = GeoVAEDataset(
                        args.mat_dir, 
                        mode=args.mode,
                        part_name=part_name)
        geo_loader = DataLoader(geo_dataset, batch_size=16, shuffle=False, num_workers=1)

        # geovae
        geo_model = get_network(geovae_config).to(args.device)
        # geo_ckpt = os.path.join(args.geovae_ckpt_dir, part_name, 'latest.pth')
        geo_ckpt = os.path.join(args.geovae_ckpt_dir, 'seat', 'latest1.pth')
        print('loading {}'.format(geo_ckpt))
        ckpt_dict = torch.load(geo_ckpt, map_location=args.device)
        geo_model = geo_model.float()
        geo_model.load_state_dict(
            ckpt_dict['model_state_dict']
            )
        geo_model.eval()

        # vqvae
        print('loading {}'.format(args.vqvae_ckpt))
        ckpt = torch.load(args.vqvae_ckpt, map_location=args.device)
        model = get_network(vqvae_config).to(args.device)
        model = model.float()
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        extract_latents_patch(geo_loader, model, geo_model, vgg16, args)
"""
jittor_code = convert(pytorch_code)
print(jittor_code)