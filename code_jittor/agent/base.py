import jittor as jt
from jittor import init
import os
from jittor import nn
from abc import abstractmethod
from tensorboardX import SummaryWriter
from util.utils import TrainClock

class BaseAgent(object):

    def __init__(self, config):
        self.log_dir = config['train']['log_dir']
        self.model_dir = config['train']['model_dir']
        self.clock = TrainClock()
        self.batch_size = config['train']['batch_size']
        self.device = config[config['mode']]['device']
        self.net = self.build_net(config)
        # self.net = self.net.to(self.device)
        self.set_loss_function()
        self.set_optimizer(config)
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        raise NotImplementedError

    def set_optimizer(self, config):
        self.optimizer = jt.optim.Adam(self.net.parameters(), config['train']['lr'])
        self.scheduler = jt.lr_scheduler.StepLR(self.optimizer, config['train']['lr_step_size'])

    def save_ckpt(self, name=None):
        if (name is None):
            save_path = os.path.join(self.model_dir, 'ckpt_epoch{}.pkl'.format(self.clock.epoch))
            print('Checkpoint saved at {}'.format(save_path))
        else:
            save_path = os.path.join(self.model_dir, '{}.pkl'.format(name))
        
        jt.save({
            'clock': self.clock.make_checkpoint(), 
            'model_state_dict': self.net.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(), 
            # 'scheduler_state_dict': self.scheduler.state_dict()
            }, save_path)
        # self.net = self.net.to(self.device)

    def load_ckpt(self, name=None):
        name = (name if (name == 'latest') else 'ckpt_epoch{}'.format(name))
        load_path = os.path.join(self.model_dir, '{}.pkl'.format(name))
        if (not os.path.exists(load_path)):
            raise ValueError('Checkpoint {} not exists.'.format(load_path))
        checkpoint = jt.load(load_path)
        print('Checkpoint loaded from {}'.format(load_path))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def execute(self, data):
        pass

    def update_network(self, loss_dict):
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        self.optimizer.backward(loss)
        self.optimizer.step()

    def update_learning_rate(self):
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[(- 1)]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for (k, v) in loss_dict.items()}
        tb = (self.train_tb if (mode == 'train') else self.val_tb)
        for (k, v) in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        self.net.train()
        (outputs, losses) = self.execute(data)
        self.update_network(losses)
        self.record_losses(losses, 'train')
        return (outputs, losses)

    def val_func(self, data):
        self.net.eval()
        with jt.no_grad():
            (outputs, losses) = self.execute(data)
        self.record_losses(losses, 'validation')
        return (outputs, losses)

    def visualize_batch(self, data, mode, **kwargs):
        raise NotImplementedError