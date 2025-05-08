
import numpy as np
import torch
import torch.nn as nn
from functools import partial

#from FCA import FCA
#from Quantizer import *
#from SSCA import SSCA
# from ..module_util import initialize_weights, flow_warp
from module_util import initialize_weights


from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
from mamba_ssm.modules.mamba_simple import Mamba


class Encoder(nn.Module):  # Lightweight Feature Encoder
    def __init__(self, in_channels, last_ch, last_kernel_w, last_stride, last_padding_w):
        super(Encoder, self).__init__()
        #self.conv = nn.Conv2d(in_channels, last_ch, (9, 4), stride=(4, 2), padding=(4, 0))
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=(3, 4), stride=(2, 2), padding=(1, 0))

        # 第二层卷积：使用小卷积核，感受野为 6x4
        self.conv2 = nn.Conv2d(128, last_ch, kernel_size=(3, last_kernel_w), stride=(last_stride, 1), padding=(1, last_padding_w))

    def forward(self, x):

        #x = self.conv(x)
        x = self.conv1(x)

        # 第二层卷积
        x = self.conv2(x)

        return x


class SSHF (nn.Module):
    def __init__(self, nf, gc=16, bias=True):
        super(SSHF , self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, (9, 1), 1, (4, 0), bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, (9, 1), 1, (4, 0), bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, (9, 1), 1, (4, 0), bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, (9, 1), 1, (4, 0), bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, (9, 1), 1, (4, 0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x





class RCMFF(nn.Module):  # Feature Enhanced Model With Attention
    def __init__(self, nf, gc):
        super(RCMFF, self).__init__()
        self.SSHF1 = SSHF(nf, gc)
        self.SSHF2 = SSHF(nf, gc)
        self.SSHF3 = SSHF(nf, gc)

    def forward(self, x):
        out = self.SSHF1(x)
        out = self.SSHF2(out)
        out = self.SSHF3(out)

        return out * 0.2 + x





class CMblock(nn.Module):  # Feature Enhanced Model With Attention
    def __init__(self, nf, gc):
        super(CMblock, self).__init__()

        self.RCMFF = RCMFF(nf, gc)

        self.Mamba = Mamba(nf,bimamba_type="v2")

    def forward(self, x):
        out = self.RCMFF(x)  # 假设 out1 的形状为 [batch_size, 96, H, W]
        b, c, h, w = out.shape
        mam = out.contiguous().view(b, h * w, c)
        mam = self.Mamba(mam)  # 假设 out2 的形状为 [batch_size, 96, H, W]
        mam = mam.view(b, c, h, w)
        out = mam + out



        return out





class ADD(nn.Module):
    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def __init__(self, in_ch, nf):
        super(ADD, self).__init__()
        block = partial(CMblock, 96,16)
        self.trunk_lb = self.make_layer(block, 8)
        self.conv_b = nn.Conv2d(in_ch, 96, 3, 1, 1, bias=True)
        self.conv_b_end = nn.Conv2d(96, 96, 3, 1, 1, groups=4, bias=True)

    def forward(self, x):
        res = self.conv_b(x)
        mam = self.conv_b(x)

        mam = self.trunk_lb(mam)
        mam = self.conv_b_end(mam)
        return mam + res


class Decoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, gc=32, up_scale=4):
        super(Decoder, self).__init__()
        self.up_scale = up_scale

        self.mafe = ADD(in_nc, nf)
        self.upconv1 = nn.Conv2d(24, 96, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(96, 192, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(48, 192, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(192, out_nc, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, x):
        fea = self.mafe(x)

        # spatial/spectral SR

        if self.up_scale == 2:
            fea = self.lrelu(self.upconv1(self.pixel_shuffle(fea)))
            fea = self.conv_last(self.lrelu(self.upconv2(fea)))
        if self.up_scale == 4:
            fea = self.lrelu(self.upconv2(self.lrelu(self.upconv1(self.pixel_shuffle(fea)))))
            fea = self.conv_last(self.lrelu(self.upconv3(self.pixel_shuffle(fea))))
        return fea

class CSN(nn.Module):
    def __init__(self, snr=0, cr=1):
        super(CSN, self).__init__()
        self.snr = snr

        if cr == 1:
            last_stride = 2
            last_ch = 27
            last_kernel_w = 1
            last_padding_w = 0
        else:
            last_stride = 1
            last_kernel_w = 2
            last_padding_w = 1

        up_scale = 4 if cr < 5 else 2

        #if cr == 1:
        #    last_ch = 27
        if cr == 5:
            last_ch = 32
        elif cr == 10:
            last_ch = 64
        elif cr == 15:
            last_ch = 103
        elif cr == 20:
            last_ch = 140

        ## 128*4*172=88064 -->
        ## 32*1*27 --> cr=1%
        ## 64*2*64 --> cr=9.30%
        ## 64*2*32 --> cr=4.65%
        ## 64*2*103--->cr=14.97%
        ## 64*2*140 -->cr=20.3%

        self.encoder = Encoder(172, last_ch, last_kernel_w, last_stride, last_padding_w)

        self.decoder = Decoder(last_ch, 172, 64, 16, up_scale=up_scale)

    def awgn(self, x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(x ** 2) / x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower



    def forward(self, data, mode=0):  # Mode=0, default, mode=1: encode only, mode=2: decoded only

        if mode == 0:
            x = self.encoder(data)
            if self.snr > 0:
                x = self.awgn(x, self.snr)
            y = self.decoder(x)
            return y, x
        elif mode == 1:
            x = self.encoder(data)
            return x
        elif mode == 2:
            return self.decoder(data)
        else:
            x = self.encoder(data)
            return self.decoder(x)


if __name__ == '__main__':
    input_data = torch.randn(1, 172, 128, 4).cuda()  # 假设batch_size为1, 输入的通道数为172，高度为32，宽度为64
    model = CSN(snr=10, cr=10).cuda()  # 初始化模型，假设cr为1

    # 进行一次前向传播来验证网络的正确性
    output, encoded = model(input_data, mode=0)  # mode=0表示编码和解码都进行
    print("Output Shape: ", output.shape)
    print("Encoded Shape: ", encoded.shape)