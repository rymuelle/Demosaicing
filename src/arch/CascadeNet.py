from arch.NAFNet import NAFBlock
import torch
import torch.nn as nn

class CascadeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=16, steps=[10, 2, 2]):
        super().__init__()
        # First step will divide image by this scale factor
        scale_factor = 2 ** (len(steps) - 1)
        self.in_convs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for step in steps:
            _width = width * scale_factor
            conv_size = scale_factor 
            stride = scale_factor
            self.in_convs.append(nn.Conv2d(in_channels, _width, kernel_size=conv_size, stride=stride))

            self.blocks.append(nn.Sequential(*[NAFBlock(_width) for _ in range(step)]))
            scale_factor = scale_factor // 2

            if _width != width:
                self.downs.append(nn.Sequential(
                    nn.Conv2d(_width, _width * 2, kernel_size=1),
                    nn.PixelShuffle(2)
                ))
            else:
                self.downs.append(nn.Identity())
        self.out = nn.Conv2d(width, out_channels, kernel_size=1)
            
    def forward(self, inp):
        output = None
        for conv, block, down in zip(self.in_convs, self.blocks, self.downs):
            x = conv(inp)
            if output is not None:
                x += output
            x = block(x)
            output = down(x)
        return self.out(output)


