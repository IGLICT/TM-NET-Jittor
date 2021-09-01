import jittor as jt
from jittor import init
from jittor import nn

class SPVAEEncoder(nn.Module):

    def __init__(self, feat_len):
        super(SPVAEEncoder, self).__init__()
        self.feat_len = feat_len
        self.mlp1 = jt.nn.Linear(feat_len, 1024)
        self.bn1 = nn.BatchNorm1d(1024, affine=None)
        self.leakyrelu1 = nn.LeakyReLU()
        self.mlp2 = jt.nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512, affine=None)
        self.leakyrelu2 = nn.LeakyReLU()
        self.mlp3 = jt.nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256, affine=None)
        self.leakyrelu3 = nn.LeakyReLU()
        self.mlp_mu = jt.nn.Linear(256, 128)
        self.sigmoid_mu = jt.nn.Sigmoid()
        self.mlp_logvar = jt.nn.Linear(256, 128)

    def execute(self, featurein):
        featureout = self.leakyrelu1(self.bn1(self.mlp1(featurein)))
        featureout = self.leakyrelu2(self.bn2(self.mlp2(featureout)))
        featureout = self.mlp3(featureout)
        mu = self.sigmoid_mu(self.mlp_mu(featureout))
        logvar = self.mlp_logvar(featureout)
        return (mu, logvar)

class SPVAEDecoder(nn.Module):

    def __init__(self, feat_len):
        super(SPVAEDecoder, self).__init__()
        self.feat_len = feat_len
        self.mlp1 = jt.nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256, affine=None)
        self.leakyrelu1 = nn.LeakyReLU()
        self.mlp2 = jt.nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512, affine=None)
        self.leakyrelu2 = nn.LeakyReLU()
        self.mlp3 = jt.nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024, affine=None)
        self.leakyrelu3 = nn.LeakyReLU()
        self.mlp4 = jt.nn.Linear(1024, self.feat_len)
        self.tanh = jt.nn.Tanh()

    def execute(self, featurein):
        featureout = self.leakyrelu1(self.bn1(self.mlp1(featurein)))
        featureout = self.leakyrelu2(self.bn2(self.mlp2(featureout)))
        featureout = self.leakyrelu3(self.bn3(self.mlp3(featureout)))
        featureout = self.tanh(self.mlp4(featureout))
        return featureout

class SPVAE(nn.Module):

    def __init__(self, geo_hidden_dim=64, part_num=7, device='cpu'):
        super(SPVAE, self).__init__()
        self.geo_hidden_dim = geo_hidden_dim
        self.part_num = part_num
        self.feat_len = (self.part_num * (((self.part_num * 2) + 9) + self.geo_hidden_dim))
        self.encoder = SPVAEEncoder(feat_len=self.feat_len)
        self.decoder = SPVAEDecoder(feat_len=self.feat_len)

    def reparameterize(self, mu, logvar):
        std = jt.exp((0.5 * logvar))
        eps = jt.randn_like(std)
        return ((eps * std) + mu)

    def execute(self, input, **kwargs):
        (mu, log_var) = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        return (z, self.decoder(z), mu, log_var)

    def loss_function(self, *args, kld_weight=0.001):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        recons_loss = jt.mse_loss(recons, input)
        kld_loss = jt.mean(((- 0.5) * jt.sum((((1 + log_var) - (mu ** 2)) - log_var.exp()), dim=1)), dim=0)
        loss = (recons_loss + (kld_weight * kld_loss))
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

if __name__ == '__main__':
    spvae = SPVAE(geo_hidden_dim=64, part_num=7)
    geo_input = jt.rand(20, 7*(2*7+9+64))
    geo_z, geo_output, mu, logvar = spvae(geo_input)