import jittor as jt
from jittor import init
import os
import scipy.io as sio
import numpy as np
from jittor import nn
from networks import get_network
from agent.base import BaseAgent

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    def execute(self, output, target):
        if self.reduction == 'mean':
            return (output-target).sqr().mean()
        elif self.reduction == 'sum':
            return (output-target).sqr().sum()

class GeoVAEAgent(BaseAgent):

    def __init__(self, config):
        super(GeoVAEAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        print(config['train']['lr'])

    def build_net(self, config):
        net = get_network(config)
        print('-----Architecture-----')
        print(net)
        return net

    def set_loss_function(self):
        self.criterion = MSELoss(reduction='mean')

    def execute(self, data):
        outputs = {}
        losses = {}
        kl_loss_weight = 0.0001
        (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, fullname) = data
        # geo_input = geo_input.to(self.device).float().contiguous()
        (geo_z, geo_output, mu, logvar) = self.net(geo_input)
        geo_recon_loss = ((self.criterion(geo_input, geo_output) * self.net.num_point) * 9)
        kl_loss = (((- 0.5) * jt.sum((((1 + logvar) - mu.pow(2)) - logvar.exp()))) / geo_output.shape[0])
        outputs['z'] = geo_z
        outputs['dec'] = geo_output
        losses['recon'] = geo_recon_loss
        losses['kl'] = (kl_loss * kl_loss_weight)
        return (outputs, losses)

    def train_func(self, data):
        self.net.train()
        (outputs, losses) = self.execute(data)
        self.update_network(losses)
        self.record_losses(losses, 'train')
        return (outputs, losses)

    def normalize_back(self, logrmax, logrmin, smax, smin, geo_output):
        logr = geo_output[:, :, :3]
        s = geo_output[:, :, 3:9]
        resultmax = 0.95
        resultmin = (- 0.95)
        s = ((((smax - smin) * (s - resultmin)) / (resultmax - resultmin)) + smin)
        logr = ((((logrmax - logrmin) * (logr - resultmin)) / (resultmax - resultmin)) + logrmin)
        geo_output = np.concatenate((logr, s), axis=2)
        return geo_output

    def visualize_batch(self, data, mode, outputs=None):
        if (mode == 'val'):
            (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, filename) = data
            filename = filename[0]
            filename = os.path.basename(filename)
            filename = filename.split('.')[0]
            # geo_input = geo_input.to(self.device).float().contiguous()
            (geo_z, geo_output, mu, logvar) = self.net(geo_input)
            origin_geo_input = origin_geo_input.data
            geo_input = geo_input.data
            geo_output = geo_output.data
            logrmax = logrmax.data
            logrmin = logrmin.data
            origin_geo_output = self.normalize_back(logrmax, logrmin, smax, smin, geo_output)
            print('{} {} {}'.format(filename, np.linalg.norm((origin_geo_input - origin_geo_output)), np.linalg.norm((geo_input - geo_output))))
            autoencode_dir = os.path.join(self.model_dir, 'autoencode'.format(mode))
            if (not os.path.exists(autoencode_dir)):
                os.mkdir(autoencode_dir)
            sio.savemat(os.path.join(autoencode_dir, (filename + '.mat')), {'geo_output': origin_geo_output}, do_compression=False)
        elif (mode == 'interpolate'):
            pass
        elif (mode == 'generate'):
            (geo_inputs, origin_geo_inputs, logrmaxs, logrmins, smaxs, smins, filenames) = data
            N = len(filenames)
            mean = outputs['z'].mean()
            std = outputs['z'].std()
            print('{} {}'.format(mean, std))
            random_z = jt.normal(mean, std, size=(N, self.net.geo_hidden_dim))
            random_outputs = self.net.geo_decoder(random_z)
            origin_geo_inputs = origin_geo_inputs.data
            geo_inputs = geo_inputs.data
            geo_outputs = random_outputs.data
            logrmaxs = logrmaxs.data
            logrmins = logrmins.data
            for i in range(N):
                origin_random_output = self.normalize_back(logrmaxs[i], logrmins[i], smaxs[i], smins[i], geo_outputs[i:(i + 1), :, :])
                generate_dir = os.path.join(self.model_dir, 'generate'.format(mode))
                if (not os.path.exists(generate_dir)):
                    os.mkdir(generate_dir)
                sio.savemat(os.path.join(generate_dir, '{}.mat'.format(i)), {'geo_output': origin_random_output}, do_compression=False)
        else:
            pass