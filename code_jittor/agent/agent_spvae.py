import jittor as jt
from jittor import init
from jittor import nn
from networks import get_network
from agent.base import BaseAgent

class SPVAEAgent(BaseAgent):

    def __init__(self, config):
        super(SPVAEAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        print(config['train']['lr'])

    def build_net(self, config):
        net = get_network(config)
        print('-----Architecture-----')
        print(net)
        net = net.to(self.device)
        return net

    def set_loss_function(self):
        self.criterion = nn.MSELoss(reduction='sum')

    def execute(self, data):
        outputs = {}
        losses = {}
        kl_loss_weight = 0.001
        (geo_input, fullname) = data
        shape = geo_input.shape
        geo_input = geo_input.reshapt((shape[0], shape[1]*shape[2]*shape[3]))
        geo_input = geo_input.to(self.device).float().contiguous()
        (geo_z, geo_output, mu, logvar) = self.net(geo_input)
        geo_recon_loss = (self.criterion(geo_input, geo_output) / geo_output.shape[0])
        kl_loss = (((- 0.5) * jt.sum((((1 + logvar) - mu.pow(2)) - logvar.exp()))) / geo_output.shape[0])
        outputs['z'] = geo_z
        outputs['dec'] = geo_output
        losses['recon'] = geo_recon_loss
        losses['kl'] = (kl_loss * kl_loss_weight)
        return (outputs, losses)

    def train_func(self, data):
        self.net.train()
        (outputs, losses) = self.forward(data)
        self.update_network(losses)
        self.record_losses(losses, 'train')
        return (outputs, losses)

    def visualize_batch(self, data, mode, outputs=None):
        pass