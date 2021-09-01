import jittor as jt
from jittor import init
import os
from jittor import nn
from networks import get_network
from agent.base import BaseAgent
from util.visualization import merge_patches

class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        pass
    def execute(self, output, target):
        if self.reduction == 'mean':
            ret = (output-target).abs().mean()
        elif self.reduction == 'sum':
            ret = (output-target).abs().sum()
        return ret

class VQVAEAgent(BaseAgent):

    def __init__(self, config):
        super(VQVAEAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        self.in_channel = config['model']['in_channel']

    def build_net(self, config):
        net = get_network(config)
        print(net)
        if config['data']['parallel']:
            net = nn.DataParallel(net)
        print(self.device)
        # net = net.to(self.device)
        return net

    def set_loss_function(self):
        self.criterion = L1Loss(reduction='sum')

    def get_seam_loss(self, recon_batches):
        if ((recon_batches.shape[0] % 6) != 0):
            print('batch size shoule be set as a multiply of 6.')
            return 0
        model_num = (recon_batches.shape[0] / 6)
        loss = 0
        for i in range(int(model_num)):
            patch0 = recon_batches[(6 * i), :, :, :]
            patch1 = recon_batches[((6 * i) + 1), :, :, :]
            patch2 = recon_batches[((6 * i) + 2), :, :, :]
            patch3 = recon_batches[((6 * i) + 3), :, :, :]
            patch4 = recon_batches[((6 * i) + 4), :, :, :]
            patch5 = recon_batches[((6 * i) + 5), :, :, :]
            loss += (
                self.criterion(patch0[:, :, 0], patch1[:, 0, :]) + \
                self.criterion(patch0[:, :, 255], jt.misc.flip(patch3[:, 0, :], [1])) + \
                self.criterion(patch0[:, 0, :], jt.misc.flip(patch4[:, 0, :], [1])) + \
                self.criterion(patch1[:, :, 0], patch4[:, :, 255]) + \
                self.criterion(patch1[:, 255, :], jt.misc.flip(patch5[:, :, 0], [1])) + \
                self.criterion(patch3[:, 255, :], patch5[:, :, 255]) + \
                self.criterion(patch4[:, 255, :], jt.misc.flip(patch5[:, 255, :], [1])) \
                )/model_num
        return loss

    def execute(self, data):
        outputs = {}
        losses = {}
        latent_loss_weight = 1
        seam_loss_weight = 0
        # img = rearrange(data[0], 'B P C H W -> (B P) C H W')
        shape = data[0].shape
        img = data[0].reshape((shape[0]*shape[1], shape[2], shape[3], shape[4]))
        filenames = data[1]
        # img = img.to(self.device).contiguous()
        # print(img.shape)
        (dec, latent_loss, quant_t, quant_b) = self.net(img)
        recon_loss = (self.criterion(dec, img) / img.shape[0])
        latent_loss = (((latent_loss.mean() * 64) * 16) * 16)
        seam_loss = self.get_seam_loss(dec)
        outputs['dec'] = dec
        losses['recon'] = recon_loss
        losses['latent'] = (latent_loss * latent_loss_weight)
        losses['seam'] = (seam_loss * seam_loss_weight)
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
        flat_filenames = []
        for i in range(len(filenames[0])):
            for j in range(len(filenames)):
                flat_filenames.append(filenames[j][i])
        filenames = flat_filenames
        recon_dir = os.path.join(self.model_dir, '{}_recon'.format(mode))
        if (not os.path.exists(recon_dir)):
            os.mkdir(recon_dir)
        dec = outputs['dec']
        dec = dec.clamp((- 1), 1)
        for i in range(dec.shape[0]):
            filename = filenames[i]
            jt.misc.save_image(dec[i, :, :, :], os.path.join(recon_dir, (filename + '.png')), nrow=1, normalize=True, range=(-1, 1))
        merge_patches(recon_dir, channel=self.in_channel)