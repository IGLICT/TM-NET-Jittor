import jittor as jt
from jittor import init
from jittor import nn
import argparse
import os
from tqdm import tqdm
from dataset.dataset_latent_geo_2levels import LatentGeo2LevelsDataset
from config import load_config
from networks import get_network
from jittor import distributions

def sample_model(model, device, batch, size, temperature, condition=None, initial_row=None):
    with jt.no_grad():
        row = jt.zeros([batch] + size, dtype='int32')
        if (initial_row is not None):
            row = initial_row
        cache = {}
        jt.gc()
        for i in tqdm(range(size[0])):
            # jt.sync_all()
            for j in range(size[1]):
                # jt.sync_all()
                # breakpoint()
                (out, cache, _) = model(row[:, :(i + 1), :], condition=condition, cache=cache)
                prob = jt.nn.softmax((out[:, :, i, j] / temperature), 1)
                m = distributions.Categorical(prob)
                # sample = m.sample().squeeze((-1))
                sample = m.sample()
                # sample = torch.multinomial(prob, 1).squeeze((- 1))
                row[:, i, j] = sample
                jt.sync_all()
                # jt.display_memory_info()
        return row

def load_model(checkpoint, config, device):
    ckpt = jt.load(os.path.join(checkpoint))
    model = get_network(config)
    model.load_parameters(ckpt['model_state_dict'])
    # model = model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    jt.flags.no_grad=1
    jt.flags.use_cuda = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--part_name', type=str, required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--vqvae_yaml', type=str, required=True)
    parser.add_argument('--top', type=str, required=True)
    parser.add_argument('--top_yaml', type=str, required=True)
    parser.add_argument('--bottom', type=str, required=True)
    parser.add_argument('--bottom_yaml', type=str, required=True)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    vqvae_config = load_config(args.vqvae_yaml)
    top_config = load_config(args.top_yaml)
    bottom_config = load_config(args.bottom_yaml)
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    
    model_vqvae = load_model(args.vqvae, vqvae_config, args.device)
    model_top = load_model(args.top, top_config, args.device)
    model_bottom = load_model(args.bottom, bottom_config, args.device)

    head_tail = os.path.split(args.top)
    head = head_tail[0]
    auto_texture_dir = os.path.join(head, 'auto_texture')
    if (not os.path.exists(auto_texture_dir)):
        os.mkdir(auto_texture_dir)
        
    dataset = LatentGeo2LevelsDataset(args.path, mode='train', part_name=args.part_name)
    loader = dataset.set_attrs(batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    ploader = tqdm(loader)
    for (i, (geo_zs, top, bottom, filenames)) in enumerate(ploader):
        # breakpoint()
        jt.sync_all()
        filename = filenames[0]
        print(filename)
        head_tail = os.path.split(filename)
        head = head_tail[0]
        basename = os.path.basename(filename)
        basename_without_ext = basename.split('.')[0]
        this_id = basename_without_ext.split('_')[0]

        # top = top.to(device)
        # bottom = bottom.to(device)
        top = top.squeeze(0)
        bottom = bottom.squeeze(0)
        # geo_zs = geo_zs.to(device)
        geo_zs = geo_zs.unsqueeze(1)
        
        for k in range(1):
            jt.sync_all()
            decoded_sample = model_vqvae.decode_code(top, bottom)
            merged_image = jt.zeros((args.batch, decoded_sample.shape[1], 768, 1024))
            H_begin = [0, 256, 256, 256, 256, 512]
            W_begin = [256, 0, 256, 512, 768, 256]
            print(decoded_sample.shape)
            for b in range(args.batch):
                for i in range(6):
                    merged_image[b, :, H_begin[i]:(H_begin[i] + 256), W_begin[i]:(W_begin[i] + 256)] = decoded_sample[((b * 6) + i), :, :, :]
                jt.misc.save_image(merged_image[b, :, :, :], os.path.join(auto_texture_dir, (((((basename_without_ext + '_') + str(k)) + '_') + str(b)) + '_sample.png')), normalize=True, range=((- 1), 1), nrow=1)
            top_sample = sample_model(model_top, device, args.batch, [96, 16], args.temp, condition=geo_zs)
            bottom_sample = sample_model(model_bottom, device, args.batch, [192, 32], args.temp, condition=top_sample)
            top_sample = top_sample.reshape((-1, 16, 16))
            # print(jt.abs((top_sample - top)).sum())
            bottom_sample = bottom_sample.reshape((-1, 32, 32))
            # print(jt.abs((bottom_sample - bottom)).sum())
            decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
            decoded_sample = decoded_sample.clamp((- 1), 1)
            merged_image = jt.zeros((args.batch, decoded_sample.shape[1], 768, 1024))
            H_begin = [0, 256, 256, 256, 256, 512]
            W_begin = [256, 0, 256, 512, 768, 256]
            
            for b in range(args.batch):
                jt.sync_all()
                jt.display_memory_info()
                for i in range(6):
                    jt.sync_all()
                    jt.display_memory_info()
                    merged_image[b, :, H_begin[i]:(H_begin[i] + 256), W_begin[i]:(W_begin[i] + 256)] = decoded_sample[((b * 6) + i), :, :, :]
                jt.misc.save_image(merged_image[b, :, :, :], os.path.join(auto_texture_dir, (((((basename_without_ext + '_') + str(k)) + '_') + str(b)) + '_sample.png')), normalize=True, range=((- 1), 1), nrow=1)
