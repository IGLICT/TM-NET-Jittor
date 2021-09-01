import jittor as jt
from jittor import init
from jittor import nn
from math import sqrt
from functools import partial, lru_cache
import numpy as np
from networks.networks_vqvae import Quantize

def wn_linear(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)

class WNConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, activation=None):
        super().__init__()
        # group = 1
        # Kh, Kw = kernel_size
        # self.weight = invariant_uniform([out_channel, in_channel//group, Kh, Kw], dtype="float")
        self.conv = jt.nn.Conv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.activation = activation

    def execute(self, input):
        out = self.conv(input)
        if (self.activation is not None):
            out = self.activation(out)
        return out

def shift_down(input, size=1):
    return jt.nn.pad(input, [0, 0, size, 0])[:, :, :input.shape[2], :]

def shift_right(input, size=1):
    return jt.nn.pad(input, [size, 0, 0, 0])[:, :, :, :input.shape[3]]

class CausalConv2d(nn.Module):

    def __init__(
        self, 
        in_channel, 
        out_channel, 
        kernel_size, 
        stride=1, 
        padding='downright', 
        activation=None
        ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = tuple([kernel_size] * 2)
        self.kernel_size = kernel_size
        if (padding == 'downright'):
            pad = tuple([(kernel_size[1] - 1), 0, (kernel_size[0] - 1), 0])

        elif ((padding == 'down') or (padding == 'causal')):
            pad = (kernel_size[1] // 2)

            pad = tuple([pad, pad, (kernel_size[0] - 1), 0])

        self.causal = 0
        if (padding == 'causal'):
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(in_channel, 
            out_channel, 
            kernel_size, 
            stride=stride, 
            padding=0, 
            activation=activation
        )

    def execute(self, input):
        out = self.pad(input)

        # if (self.causal > 0):
        #     self.conv.conv.weight[:, :, -1, self.causal:] = 0

        out = self.conv(out)

        return out

class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self,x):
        half = x.shape[1]//2
        return x[:, :half, :, :]*jt.sigmoid(x[:, half:, :, :])

class GatedResBlock(nn.Module):

    def __init__(self, 
                in_channel, 
                channel, 
                kernel_size, 
                conv='wnconv2d', 
                activation=nn.Sigmoid, 
                dropout=0.1, 
                auxiliary_channel=0, 
                condition_dim=0):
        super().__init__()

        if (conv == 'wnconv2d'):
            conv_module = partial(WNConv2d, padding=(kernel_size // 2))

        elif (conv == 'causal_downright'):
            conv_module = partial(CausalConv2d, padding='downright')

        elif (conv == 'causal'):
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        
        self.conv1 = conv_module(in_channel, channel, kernel_size)
        
        if (auxiliary_channel > 0):
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = conv_module(channel, (in_channel * 2), kernel_size)

        if (condition_dim > 0):
            self.condition = WNConv2d(condition_dim, (in_channel * 2), 1, bias=False)

        self.gate = GLU()

    def execute(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if (aux_input is not None):
            out = (out + self.aux_conv(self.activation(aux_input)))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if (condition is not None):
            condition = self.condition(condition)
            out += condition
        out = self.gate(out)
        out += input
        return out

@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0
    return (
        jt.Var(mask).unsqueeze(0), 
        jt.Var(start_mask).unsqueeze(1)
    )

def repeat(x, *shape):
    r'''
    Repeats this var along the specified dimensions.

    Args:

        x (var): jittor var.

        shape (tuple): int or tuple. The number of times to repeat this var along each dimension.
 
    Example:

        >>> x = jt.array([1, 2, 3])

        >>> x.repeat(4, 2)
        [[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]]

        >>> x.repeat(4, 2, 1).size()
        [4, 2, 3,]
    '''
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    len_x_shape = len(x.shape)
    len_shape = len(shape)
    x_shape = x.shape
    rep_shape = shape
    if len_x_shape < len_shape:
        x_shape = (len_shape - len_x_shape) * [1] + x.shape
        x = x.broadcast(x_shape)
    elif len_x_shape > len_shape:
        rep_shape = (len_x_shape - len_shape) * [1] + list(shape)
    #TODO if input.shape[i]=1, no add [1]
    reshape_shape = []
    broadcast_shape = []
    for x_s,r_s in zip(x_shape,rep_shape):
        reshape_shape.append(1)
        reshape_shape.append(x_s)

        broadcast_shape.append(r_s)
        broadcast_shape.append(1)

    x = x.reshape(reshape_shape)
    x = x.broadcast(broadcast_shape)

    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()

    x = x.reshape(tar_shape)
    return x


class CausalAttention(nn.Module):
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def execute(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose((0, 2, 1, 3))

        query_flat = query.view(batch, query.shape[1], -1).transpose((0, 2, 1))
        key_flat = key.view(batch, key.shape[1], -1).transpose((0, 2, 1))
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose((0, 1, 3, 2))
        value = reshape(self.value(key_flat))

        attn = jt.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        # attn = attn.masked_fill(mask == 0, -1e4)
        
        mask1 = mask.unsqueeze(0)
        mask1 = repeat(mask1, attn.shape[0], attn.shape[1])
        mask1 = (mask1 == 0)
        attn[mask1] = -1e4
        attn = jt.nn.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(0, 2, 1, 3).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def execute(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = jt.concat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = jt.concat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = jt.concat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def execute(self, input):
        return self.blocks(input)

def one_hot(index, n_class):
    ret = jt.zeros((index.shape[0], n_class))
    index1 = jt.arange(0, index.shape[0]).reshape(-1, 1)
    index2 = index.reshape(-1, 1)
    ret[index1, index2] = 1
    return ret

class PixelSNAILTop(nn.Module):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
        n_condition_dim=64,
        n_condition_class=256,
        n_condition_sub_dim=16,
        n_default_condition_dim=64,
    ):
        super().__init__()
        height, width = shape

        self.shape = shape
        self.n_class = n_class
        self.n_condition_dim = n_condition_dim
        self.n_condition_class = n_condition_class
        self.n_condition_sub_dim = n_condition_sub_dim
        self.n_default_condition_dim = n_default_condition_dim
        
        if self.n_condition_dim > self.n_default_condition_dim:
            self.linear1 = jt.nn.Linear(self.n_condition_dim, 256)
            self.bn1 = nn.BatchNorm1d(num_features=256)
            self.relu1 = jt.nn.ReLU()
            self.linear2 = jt.nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(num_features=128)
            self.relu2 = jt.nn.ReLU()
            self.linear3 = jt.nn.Linear(128, 64)
            self.relu3 = jt.nn.ReLU()
            
        self.quantize = Quantize(self.n_condition_sub_dim, self.n_condition_class)

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size


        self.horizontal = CausalConv2d(
            n_class, channel, (kernel // 2, kernel), padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, ((kernel + 1) // 2, kernel // 2), padding='downright'
        )

        coord_x = (jt.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (jt.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.background = jt.concat([coord_x, coord_y], 1)

        self.blocks = nn.ModuleList()
        
        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                self.n_condition_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.Sigmoid(), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def execute(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = input.reshape([batch*height*width])
        input = (
            one_hot(input, self.n_class).reshape([batch, height, width, self.n_class]).permute(0, 3, 1, 2).type_as(self.background)
        )
        # input = input.contiguous()
        horizontal = shift_down(self.horizontal(input))
        # horizontal = horizontal.contiguous()
        vertical = shift_right(self.vertical(input))
        # vertical = vertical.contiguous()
        out = horizontal + vertical
        # out = out.contiguous()

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                if self.n_condition_dim > self.n_default_condition_dim:
                    # print(condition.shape)
                    condition = condition.squeeze(1)
                    condition = condition.squeeze(1)
                    # print(condition.shape)
                    condition = self.relu1(self.bn1(self.linear1(condition)))
                    condition = self.relu2(self.bn2(self.linear2(condition)))
                    condition = self.relu3(self.linear3(condition))
                    condition = condition.unsqueeze(1)
                    condition = condition.unsqueeze(1)
                temp_shape = condition.shape
                reshaped_condition = condition.reshape(
                        (
                        temp_shape[0], 
                        temp_shape[1], 
                        temp_shape[2]*temp_shape[3]//self.n_condition_sub_dim, 
                        self.n_condition_sub_dim
                        )
                    )
                # reshaped_condition = einops.rearrange(condition, 'B H W (T C) -> B H (W T) C', T=self.n_default_condition_dim//self.n_condition_sub_dim)
                # print(reshaped_condition.shape)
                _, latent_diff, condition_id = self.quantize(reshaped_condition)
                self.latent_diff = latent_diff

                repeated_condition_id = condition_id.repeat(1, 
                                                    self.shape[0]//condition_id.shape[1]//2, 
                                                    self.shape[1]//condition_id.shape[2]//2)
                # repeated_condition_id = repeated_condition_id.to(torch.long)
                batch1, height1, width1 = repeated_condition_id.shape
                repeated_condition_id = repeated_condition_id.reshape([batch1*height1*width1])
                condition = (
                    one_hot(repeated_condition_id, self.n_condition_class)
                    .reshape([batch1, height1, width1, self.n_condition_class])
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                condition = jt.nn.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]
            # condition = condition.contiguous()
            
        for block in self.blocks:
            out = block(out, background, condition=condition)
            # out = out.contiguous()
        # out = out.contiguous()
        out = self.out(out)
        # out = out.contiguous()
        
        return out, cache, self.latent_diff


class PixelSNAIL(nn.Module):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
        n_condition_class=0
    ):
        super().__init__()
        height, width = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size
        if n_condition_class == 0:
            n_condition_class = n_class
        self.n_condition_class = n_class

        self.horizontal = CausalConv2d(
            n_class, channel, (kernel // 2, kernel), padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, ((kernel + 1) // 2, kernel // 2), padding='downright'
        )

        coord_x = (jt.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (jt.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.background = jt.concat([coord_x, coord_y], 1)

        self.blocks = nn.ModuleList()
        
        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                self.n_condition_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.Sigmoid(), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def execute(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = input.reshape([batch*height*width])
        input = (
            one_hot(input, self.n_class).reshape([batch, height, width, self.n_class]).permute(0, 3, 1, 2).type_as(self.background)
        )
        # input = input.contiguous()
        horizontal = shift_down(self.horizontal(input))
        # horizontal = horizontal.contiguous()
        vertical = shift_right(self.vertical(input))
        # vertical = vertical.contiguous()
        out = horizontal + vertical
        # out = out.contiguous()

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                batch1, height1, width1 = condition.shape
                condition = condition.reshape([batch1*height1*width1]) 
                condition = (
                    one_hot(condition, self.n_condition_class)
                    .reshape([batch1, height1, width1, self.n_condition_class])
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                condition = jt.nn.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]
            # condition = condition.contiguous()
            
        for block in self.blocks:
            out = block(out, background, condition=condition)
            # out = out.contiguous()
        out = self.out(out)
        
        return out, cache, None

if __name__ == '__main__':
    pixelsnail_top = PixelSNAILTop(
        shape=[16,16],
        n_class=256,
        channel=256,
        kernel_size=5 ,
        n_block=4,
        n_res_block=4,
        res_channel=128,
        attention=True,
        dropout=0.1,
        n_cond_res_block=3,
        cond_res_channel=256,
        cond_res_kernel=3,
        n_out_res_block=0,
        n_condition_dim=64,
        n_condition_class=256,
        n_condition_sub_dim=16,
    )
    top = jt.rand((10, 16, 16))
    geo_zs = jt.rand((10, 1, 1, 64))
    target = top
    out, _, latent_diff = pixelsnail_top(top, condition=geo_zs)

    pixelsnail = PixelSNAIL(
        shape=[32,32],
        n_class=256,
        channel=256,
        kernel_size=5 ,
        n_block=4,
        n_res_block=4,
        res_channel=128,
        attention=True,
        dropout=0.1,
        n_cond_res_block=3,
        cond_res_channel=256,
        cond_res_kernel=3,
        n_out_res_block=0,
        n_condition_class=0
    )
    top = jt.rand((10, 16, 16))
    bottom = jt.rand((10, 32, 32))
    
    out, _, latent_diff = pixelsnail(bottom, condition=top)