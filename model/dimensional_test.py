import torch
import torch.nn as nn
from utils import tools

input = torch.rand(30, 1, 104, 35) # (B, C, T, F)

conv1 = nn.Conv2d(in_channels=1,
                    out_channels=128,
                    kernel_size=[5,15],
                    stride=1,
                    padding=(2, 7))
x = conv1(input)
print(x.shape)

conv2 = nn.Conv2d(in_channels=128,
                    out_channels=256,
                    kernel_size=[5,5],
                    stride=2,
                    padding=(2, 2))
x = conv2(x)
print(x.shape)

conv3 = nn.Conv2d(in_channels=256,
                    out_channels=512,
                    kernel_size=[5,5],
                    stride=2,
                    padding=(2, 2))
x = conv3(x)
print(x.shape)

x = x.view(30, -1, int(104/4), 1)
print(x.shape)

conv4 = nn.Conv2d(in_channels=4608,
                    out_channels=256,
                    kernel_size=[1,1],
                    stride=1)
x = conv4(x)
print(x.shape)

conv5 = nn.Conv2d(in_channels=256,
                    out_channels=512,
                    kernel_size=[3,1],
                    stride=1,
                    padding=(1, 0))
x = conv5(x)
print(x.shape)

conv6 = nn.Conv2d(in_channels=512,
                    out_channels=256,
                    kernel_size=[3,1],
                    stride=1,
                    padding=(1, 0))
x = conv6(x)
print(x.shape)

conv7 = nn.Conv2d(in_channels=256,
                    out_channels=4608,
                    kernel_size=[1,1],
                    stride=1,)
x = conv7(x)
print(x.shape)

x = x.view(30, -1, int(104/4), 9)
print(x.shape)

conv8 = nn.Conv2d(in_channels=512,
                    out_channels=1024,
                    kernel_size=[5,5],
                    stride=1,
                    padding=(2, 2))
x = conv8(x)
print(x.shape)

ps1 = nn.PixelShuffle(2)
x = ps1(x)
print(x.shape)

conv9 = nn.Conv2d(in_channels=256,
                    out_channels=512,
                    kernel_size=[5,5],
                    stride=1,
                    padding=(2, 2))
x = conv9(x)
print(x.shape)

ps1 = nn.PixelShuffle(2)
x = ps1(x)
print(x.shape)

conv10 = nn.Conv2d(in_channels=128,
                    out_channels=1,
                    kernel_size=[5,15],
                    stride=1,
                    padding=(2, 7))
x = conv10(x)
print(x.shape)
tools.tensor_to_img(x[0].squeeze(0))
