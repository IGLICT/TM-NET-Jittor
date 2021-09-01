import jittor as jt
from jittor import init
from jittor import nn
import math

def one_hot(index, n_class):
    ret = jt.zeros((index.shape[0], n_class))
    index1 = jt.arange(0, index.shape[0]).reshape(-1, 1)
    index2 = index.reshape(-1, 1)
    ret[index1, index2] = 1
    return ret

class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = jt.rand(dim, n_embed)
        self.embed = embed
        self.cluster_size = jt.zeros(n_embed)
        self.embed_avg = embed.clone()

    def execute(self, input):
        flatten = input.reshape((-1, self.dim))
        dist = ((flatten.pow(2).sum(1, keepdims=True) - ((2 * flatten) @ self.embed)) + self.embed.pow(2).sum(0, keepdims=True))
        # (_, embed_ind) = (- dist).max(1)
        (embed_ind, _) = (-dist).argmax(1)
        embed_onehot = one_hot(embed_ind, self.n_embed)
        embed_ind = embed_ind.view(*input.shape[:(- 1)])
        quantize = self.embed_code(embed_ind)
        if self.is_training():
            self.cluster_size = self.cluster_size.multiply(self.decay).add(embed_onehot.sum(0).multiply(1 - self.decay))
            embed_sum = (flatten.transpose() @ embed_onehot)
            self.embed_avg = self.embed_avg.multiply(self.decay).add(embed_sum.multiply(1 - self.decay))
            n = self.cluster_size.sum()
            cluster_size = (((self.cluster_size + self.eps) / (n + (self.n_embed * self.eps))) * n)
            embed_normalized = (self.embed_avg / cluster_size.unsqueeze(0))
            self.embed = embed_normalized.clone()
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = (input + (quantize - input).detach())
        return (quantize, diff, embed_ind)

    def embed_code(self, embed_id):
        # return F.embedding(embed_id, self.embed.transpose(0, 1))
        shape = embed_id.shape
        shape.append(self.embed.shape[0])
        embed_id = embed_id.reshape(-1, 1)
        ret = self.embed.transpose()[embed_id, :]
        ret = jt.reshape(ret, shape)
        return ret

class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.ReLU(), 
                        nn.Conv(in_channel, channel, 3, padding=1), 
                        nn.ReLU(), 
                        nn.Conv(channel, in_channel, 1)
                    )

    def execute(self, input):
        out = self.conv(input)
        out += input
        return out

class Encoder(nn.Module):

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if (stride == 4):
            blocks = [
                nn.Conv(in_channel, (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 2), channel, 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv(channel, channel, 3, padding=1)
                ]
        elif (stride == 2):
            blocks = [
                nn.Conv(in_channel, (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 2), channel, 3, padding=1)
                ]
        elif (stride == 8):
            blocks = [
                nn.Conv(in_channel, (channel // 4), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 4), (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), nn.Conv((channel // 2), channel, 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv(channel, channel, 3, padding=1)
                ]
        elif (stride == 16):
            blocks = [
                nn.Conv(in_channel, (channel // 8), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 8), (channel // 4), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 4), (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv((channel // 2), channel, 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.Conv(channel, channel, 3, padding=1)
                ]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

    def execute(self, input):
        return self.blocks(input)

class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU())
        if (stride == 4):
            blocks.extend([
                nn.ConvTranspose(channel, (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 2), out_channel, 4, stride=2, padding=1)
                ])
        elif (stride == 2):
            blocks.append(nn.ConvTranspose(channel, out_channel, 4, stride=2, padding=1))
        elif (stride == 8):
            blocks.extend([
                nn.ConvTranspose(channel, (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 2), (channel // 4), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 4), out_channel, 4, stride=2, padding=1)
                ])
        elif (stride == 16):
            blocks.extend([
                nn.ConvTranspose(channel, (channel // 2), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 2), (channel // 4), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 4), (channel // 8), 4, stride=2, padding=1), 
                nn.ReLU(), 
                nn.ConvTranspose((channel // 8), out_channel, 4, stride=2, padding=1)
                ])
        self.blocks = nn.Sequential(*blocks)

    def execute(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):

    def __init__(self, in_channel=4, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=256, stride=4):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv((embed_dim + channel), embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder((embed_dim + embed_dim), in_channel, channel, n_res_block, n_res_channel, stride=stride)

    def execute(self, input):
        (quant_t, quant_b, diff, id_t, id_b) = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        return (dec, diff, quant_t, quant_b)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute((0, 2, 3, 1))
        (quant_t, diff_t, id_t) = self.quantize_t(quant_t)
        quant_t = quant_t.permute((0, 3, 1, 2))
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t)
        enc_b = jt.contrib.concat([dec_t, enc_b], dim=1)
        quant_b = self.quantize_conv_b(enc_b).permute((0, 2, 3, 1))
        (quant_b, diff_b, id_b) = self.quantize_b(quant_b)
        quant_b = quant_b.permute((0, 3, 1, 2))
        diff_b = diff_b.unsqueeze(0)
        return (quant_t, quant_b, (diff_t + diff_b), id_t, id_b)

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = jt.contrib.concat([upsample_t, quant_b], dim=1)
        dec = self.dec(quant)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute((0, 3, 1, 2))
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute((0, 3, 1, 2))
        dec = self.decode(quant_t, quant_b)
        return dec

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    vqvae = VQVAE()
    img = jt.rand((2, 4, 256, 256))
    (dec, diff, quant_t, quant_b) = vqvae(img)
    print(dec.shape)
