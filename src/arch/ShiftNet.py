# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialShift(nn.Module):
    def __init__(self, channels, shift_fraction=0.25):
        """
        Args:
            channels (int): Total number of input channels (C).
            shift_fraction (float): Total percentage of channels to shift. 
                                    Shared equally among the 4 directions.
        """
        super().__init__()
        
        self.shift_channels = int(channels * shift_fraction) // 4
        self.id_channels = channels - (self.shift_channels * 4)
        
        self.split_sizes = [
            self.shift_channels, # Up
            self.shift_channels, # Down
            self.shift_channels, # Left
            self.shift_channels, # Right
            self.id_channels     # Identity
        ]

    def forward(self, x):
        # x shape: (N, C, H, W)
        
        out_up, out_down, out_left, out_right, out_id = torch.split(
            x, self.split_sizes, dim=1
        )
        out_up = F.pad(out_up[:, :, 1:, :], (0, 0, 0, 1))
        out_down = F.pad(out_down[:, :, :-1, :], (0, 0, 1, 0))
        out_left = F.pad(out_left[:, :, :, 1:], (0, 1, 0, 0))
        out_right = F.pad(out_right[:, :, :, :-1], (1, 0, 0, 0))
        return torch.cat([out_up, out_down, out_left, out_right, out_id], dim=1)
    

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class ShifterBlock(nn.Module):
    def __init__(self, c, expand=2, kernel_size=1, padding=0, groups=1):
        super().__init__()
        self.sg = SimpleGate()
        self.norm1 = nn.GroupNorm(1, c)
        expand = expand * c
        self.shifter = SpatialShift(c, shift_fraction=0.5)
        self.conv0 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=kernel_size, padding=padding, stride=1, groups=groups, bias=True)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=expand, kernel_size=kernel_size, padding=padding, stride=1, groups=groups, bias=True)
        self.conv2 = nn.Conv2d(in_channels=expand // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=groups, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.shifter(x)
        x = self.conv1(self.norm1(x))
        x = self.sg(x)
        x = self.conv2(x)
        return inp + x * self.beta
    
class NAFBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        self.shifter = SpatialShift(c, shift_fraction=0.5)
        self.shift_pw = nn.Conv2d(c, c, 1)

        self.pblock = MixerBlock(c, kernel_size=1, padding=0, expand=expand, groups=1)
        # self.pblock2 = MixerBlock(c, kernel_size=1, padding=0, expand=expand, groups=1)

    def forward(self, x):
        inp = x
        x = self.shift_pw(x)
        x = self.shifter(x)
        x = self.pblock(x)
        # x = inp + x
        # x = self.pblock2(x)
        return x + inp

class SimpleBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2):
        super().__init__()
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = nn.GroupNorm(1, c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.conv1(self.norm1(x))
        x = self.sg(x)
        return inp + x * self.beta

        return y + x * self.gamma
class ShiftNet(nn.Module):

    def __init__(self, img_channel=3, in_channels=6, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], mask=True):
        super().__init__()
 
        img_channel = in_channels // 2
        self.img_channel = img_channel
        if in_channels is None:
            in_channels = img_channel
        self.mask = mask

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
 
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num, snum in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[ShifterBlock(chan) for _ in range(num)],
                    *[SimpleBlock(chan) for _ in range(snum)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[ShifterBlock(chan) for _ in range(middle_blk_num)]
            )

        for num, snum in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[ShifterBlock(chan) for _ in range(num)],
                    *[SimpleBlock(chan) for _ in range(snum)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x[:, :, :H, :W]

        if self.mask:
            sparse, mask = inp.chunk(2, dim=1)
            x = x * (mask==0) + sparse

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



