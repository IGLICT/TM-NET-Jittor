import jittor as jt
from jittor import init
from jittor import nn
from jittor_geometric import nn as gnn
import scipy.io as sio
import numpy as np

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    def execute(self, output, target):
        if self.reduction == 'mean':
            return (output-target).sqr().mean()
        elif self.reduction == 'sum':
            return (output-target).sqr().sum()

class PartDeformEncoder(nn.Module):

    def __init__(self, num_point, feat_len, edge_index=None, probabilistic=True, bn=True):
        super(PartDeformEncoder, self).__init__()
        self.probabilistic = probabilistic
        self.bn = bn
        self.edge_index = edge_index
        self.conv1 = gnn.GCNConvMod(9, 9, edge_index)
        self.conv2 = gnn.GCNConvMod(9, 9, edge_index)
        self.conv3 = gnn.GCNConvMod(9, 9, edge_index)
        if self.bn:
            self.bn1 = jt.nn.InstanceNorm1d(9)
            self.bn2 = jt.nn.InstanceNorm1d(9)
            self.bn3 = jt.nn.InstanceNorm1d(9)
        self.mlp2mu = nn.Linear((num_point * 9), feat_len)
        self.mlp2var = nn.Linear((num_point * 9), feat_len)

    def execute(self, featurein):
        feature = featurein
        self.vertex_num = feature.shape[1]
        if self.bn:
            net = jt.nn.leaky_relu(self.bn1(self.conv1(feature)), scale=0)
            net = jt.nn.leaky_relu(self.bn2(self.conv2(net)), scale=0)
            net = jt.nn.leaky_relu(self.bn3(self.conv3(net)), scale=0)
        else:
            net = jt.nn.leaky_relu(self.conv1(feature), scale=0)
            net = jt.nn.leaky_relu(self.conv2(net), scale=0)
            net = jt.nn.leaky_relu(self.conv3(net), scale=0)
        net = net.reshape((-1, (self.vertex_num * 9)))
        mu = self.mlp2mu(net)
        logvar = self.mlp2var(net)
        return (mu, logvar)

class PartDeformDecoder(nn.Module):

    def __init__(self, feat_len, num_point, edge_index=None, bn=True):
        super(PartDeformDecoder, self).__init__()
        self.num_point = num_point
        self.bn = bn
        self.mlp1 = nn.Linear(feat_len, (self.num_point * 9), bias=False)
        self.conv1 = gnn.GCNConvMod(9, 9, edge_index)
        self.conv2 = gnn.GCNConvMod(9, 9, edge_index)
        self.conv3 = gnn.GCNConvMod(9, 9, edge_index)
        if bn:
            self.bn1 = jt.nn.InstanceNorm1d(9)
            self.bn2 = jt.nn.InstanceNorm1d(9)
            self.bn3 = jt.nn.InstanceNorm1d(9)
        self.L2Loss = MSELoss(reduction='mean')

    def execute(self, net):
        net = self.mlp1(net).view(((- 1), self.num_point, 9))
        if self.bn:
            net = jt.nn.leaky_relu(self.bn1(self.conv1(net)), scale=0)
            net = jt.nn.leaky_relu(self.bn2(self.conv2(net)), scale=0)
            net = jt.nn.leaky_relu(self.bn3(self.conv3(net)), scale=0)
        else:
            net = jt.nn.leaky_relu(self.conv1(net), scale=0)
            net = jt.nn.leaky_relu(self.conv2(net), scale=0)
        net = jt.tanh(self.conv3(net))
        return net

    def loss(self, pred, gt):
        avg_loss = (self.L2Loss(pred, gt) * 100000)
        return avg_loss

class GeoVAE(nn.Module):

    def __init__(self, 
                geo_hidden_dim=128, 
                ref_mesh_mat='../guitar_with_mapping.mat', 
                device='cpu'):
        super(GeoVAE, self).__init__()
        self.geo_hidden_dim = geo_hidden_dim
        self.ref_mesh_mat = ref_mesh_mat
        # self.device = torch.device(device)
        ref_mesh_data = sio.loadmat(self.ref_mesh_mat)
        V = ref_mesh_data['V']
        F = ref_mesh_data['F']
        edge_index = ref_mesh_data['edge_index'].astype(np.int64).transpose()
        edge_index = jt.float32(edge_index)
        self.num_point = V.shape[0]
        print(self.num_point)
        self.geo_encoder = PartDeformEncoder(self.num_point, self.geo_hidden_dim, edge_index, probabilistic=False, bn=False)
        self.geo_decoder = PartDeformDecoder(self.geo_hidden_dim, self.num_point, edge_index, bn=False)

    def encode(self, geo_input):
        (mu, logvar) = self.geo_encoder(geo_input)
        # (mu, logvar) = (mu.contiguous(), logvar.contiguous())
        return (mu, logvar)

    def decode(self, geo_z):
        geo_output = self.geo_decoder(geo_z)
        return geo_output

    def reparameterize(self, mu, logvar):
        std = jt.exp((0.5 * logvar))
        eps = jt.randn_like(std)
        return (mu + (eps * std))

    def execute(self, geo_input):
        (mu, logvar) = self.encode(geo_input)
        geo_z = self.reparameterize(mu, logvar)
        geo_output = self.decode(geo_z)
        return (geo_z, geo_output, mu, logvar)

if __name__ == '__main__':
    geovae = GeoVAE(128, '/mnt/f/wutong/data/table/table_std.mat', device='cpu')
    geo_input = jt.rand(20, geovae.num_point, 9)
    geo_z, geo_output, mu, logvar = geovae(geo_input)
