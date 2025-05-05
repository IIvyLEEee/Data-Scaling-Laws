import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import quant_module.quant_conv1d as quant_conv1d
import quant_module.quant_convtranspose1d as quant_convtranspose1d

# from einops.layers.torch import Rearrange

class Downsample1d(nn.Module):
    def __init__(self, dim, quant_layer=False):
        super().__init__()
        if quant_layer:
            self.conv = quant_conv1d.QuantConv1d(dim, dim, 3, 2, 1)
        else:
            self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim, quant_layer=False):
        super().__init__()
        if quant_layer:
            self.conv = quant_convtranspose1d.QuantConvTranspose1d(dim, dim, 4, 2, 1)
        else:
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, quant_layer=False):
        super().__init__()

        if quant_layer:
            self.block = nn.Sequential(
                quant_conv1d.QuantConv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    #x = torch.zeros((1,256,16))
    x = torch.randn((1,256,16))
    print(x)
    o = cb(x)
    print("输出o的shape:", o.shape)
    print("输出o的均值:", o.mean().item())
    print("输出o的最大值:", o.max().item())
    print("输出o的最小值:", o.min().item())
    print("输出o的内容:", o)

if __name__ == "__main__":
    test()