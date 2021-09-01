import jittor as jt
from jittor import init
import os
from jittor import nn
from networks import get_network
from agent.base import BaseAgent

class PixelSNAILAgent(BaseAgent):

    def __init__(self, config):
        super(PixelSNAILAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        self.hier = config['model']['name'].split('_')[1]
        print(config['train']['lr'])

    def build_net(self, config):
        net = get_network(config)
        print('-----Architecture-----')
        print(net)
        return net

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss()

    def execute(self, data):
        outputs = {}
        losses = {}
        (geo_zs, top, bottom, filenames) = data
        geo_zs = geo_zs.unsqueeze(1)
        top = top.reshape((top.shape[0], (top.shape[1] * top.shape[2]), top.shape[3]))
        bottom = bottom.reshape((bottom.shape[0], (bottom.shape[1] * bottom.shape[2]), bottom.shape[3]))
        if (self.hier == 'top'):
            # top = top.to(self.device).contiguous()
            # geo_zs = geo_zs.to(self.device).contiguous()
            target = top
            (out, _, latent_diff) = self.net(top, condition=geo_zs)
        elif (self.hier == 'bottom'):
            # top = top.to(self.device).contiguous()
            # bottom = bottom.to(self.device).contiguous()
            target = bottom
            (out, _, latent_diff) = self.net(bottom, condition=top)
        CE_loss = self.criterion(out, target)
        (pred, _) = out.argmax(1)
        correct = (pred == target).float()
        accuracy = (correct.sum() / target.numel())
        print(accuracy)
        outputs['out'] = out
        losses['CE'] = CE_loss
        if (latent_diff is not None):
            losses['latent'] = latent_diff
        return (outputs, losses)

    def train_func(self, data):
        self.net.train()
        (outputs, losses) = self.execute(data)
        self.update_network(losses)
        self.record_losses(losses, 'train')
        return (outputs, losses)

    def visualize_batch(self, data, mode, outputs=None):
        if (mode == 'train'):
            return
        imgs = data[0]
        filenames = data[1]
        recon_dir = os.path.join(self.model_dir, '{}_recon'.format(mode))
        if (not os.path.exists(recon_dir)):
            os.mkdir(recon_dir)
        pass