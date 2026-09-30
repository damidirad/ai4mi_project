#!/usr/bin/env python3.10
# ENet with 2.5d: the initial block and bottleneck1 run in 3D, the rest in 2D on the center slice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


def conv_block3d(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv3d(in_dim, out_dim, **kwconv),
                             nn.BatchNorm3d(out_dim),
                             nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(kernel_size, 1),
                                       padding=(kernel_size // 2, 0)),
                             nn.Conv2d(out_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       padding=(0, kernel_size // 2)),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)
                if not asym:
                        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                else:
                        self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

                if in_dim > out_dim:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
                elif dilate_last:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
                else:
                        self.conv_out = nn.Identity()

        def forward(self, in_) -> Tensor:
                do = self.do(self.block2(self.block1(self.block0(in_))))
                return self.PReLU_out(self.conv_out(in_) + do)


class BottleNeck3d(nn.Module):
        def __init__(self, dim, projectionFactor, *, dropoutRate=0.01):
                super().__init__()
                mid_dim: int = dim // projectionFactor

                self.block0 = conv_block3d(dim, mid_dim, kernel_size=1)
                self.block1 = conv_block3d(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block3d(mid_dim, dim, kernel_size=1)

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

        def forward(self, in_) -> Tensor:
                do = self.do(self.block2(self.block1(self.block0(in_))))
                return self.PReLU_out(in_ + do)


class BottleNeckDownSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

                self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

        def forward(self, in_) -> tuple[Tensor, Tensor]:
                maxpool_output, indices = self.maxpool0(in_)

                output = self.do(self.block2(self.block1(self.block0(in_))))
                c = maxpool_output.shape[1]
                output[:, :c] += maxpool_output

                return self.PReLU(output), indices


class BottleNeckDownSampling3d(nn.Module):
        # downsample only h,w and preserve depth
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                self.maxpool0 = nn.MaxPool3d((1, 2, 2), return_indices=True)

                self.block0 = conv_block3d(in_dim, mid_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.block1 = conv_block3d(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block3d(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

        def forward(self, in_) -> tuple[Tensor, Tensor]:
                maxpool_output, indices = self.maxpool0(in_)

                output = self.do(self.block2(self.block1(self.block0(in_))))
                c = maxpool_output.shape[1]
                output[:, :c] += maxpool_output

                return self.PReLU(output), indices


class BottleNeckUpSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                self.unpool = nn.MaxUnpool2d(2)

                self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

        def forward(self, args) -> Tensor:
                in_, indices, skip = args

                up = self.unpool(in_, indices)
                do = self.do(self.block2(self.block1(self.block0(torch.cat((up, skip), dim=1)))))

                return self.PReLU(up + do)


class ENet_2_5d(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                factor: int = kwargs.get("factor", 4)  # projection factor
                K: int = kwargs.get("kernels", 16)  # n_kernels
                self.in_dim = in_dim

                # 3d operations on the input volume
                self.conv0 = nn.Conv3d(in_dim, K - in_dim, kernel_size=3, stride=(1, 2, 2), padding=1)
                self.maxpool0 = nn.MaxPool3d((1, 2, 2))

                # downsampling half
                self.bottleneck1_0 = BottleNeckDownSampling3d(K, K * 4, factor)
                self.bottleneck1_1 = nn.Sequential(BottleNeck3d(K * 4, factor),
                                                   BottleNeck3d(K * 4, factor),
                                                   BottleNeck3d(K * 4, factor),
                                                   BottleNeck3d(K * 4, factor))

                # 2d on the center slice
                self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, factor)
                self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, factor, dilation=2),
                                                   BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, factor, dilation=4),
                                                   BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, factor, dilation=8),
                                                   BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, factor, dilation=16))

                # 2d operations, middle of the network
                self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, factor, dilation=2),
                                                 BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 8, factor, dilation=4),
                                                 BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, factor, dilation=8),
                                                 BottleNeck(K * 8, K * 8, factor, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 4, factor, dilation=16, dilate_last=True))

                # upsampling half
                self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, factor),
                                                 BottleNeck(K * 4, K * 4, factor, dropoutRate=0.1),
                                                 BottleNeck(K * 4, K, factor, dropoutRate=0.1))
                self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, factor),
                                                 BottleNeck(K, K, factor, dropoutRate=0.1))

                # fn upsampling and convolutions
                self.final = nn.Sequential(conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           nn.Conv2d(K, out_dim, kernel_size=1))

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def _as_volume(self, input: Tensor) -> Tensor:
                # (B, slices * in_dim, H, W) -> (B, in_dim, slices, H, W)
                if input.dim() == 5:
                        return input

                b, c, h, w = input.shape
                assert c % self.in_dim == 0, f"Cannot split {c} channels into {self.in_dim} per slice"
                return input.reshape(b, c // self.in_dim, self.in_dim, h, w).transpose(1, 2)

        def forward(self, input):
                volume = self._as_volume(input)
                d = volume.shape[2] // 2  # Center slice

                # 3d operations on the input volume
                initial = torch.cat((self.conv0(volume), self.maxpool0(volume)), dim=1)

                # downsampling half, first stage in 3d
                bn1_0, indices_3d = self.bottleneck1_0(initial)
                bn1_3d = self.bottleneck1_1(bn1_0)

                # collapse to the center slice, shift back to 2d indices for MaxUnpool2d.
                h, w = initial.shape[-2:]
                outputInitial = initial[:, :, d]
                bn1_out = bn1_3d[:, :, d]
                indices_1 = indices_3d[:, :, d] - d * h * w

                # rest of network in 2d on the center slice
                bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
                bn2_out = self.bottleneck2_1(bn2_0)

                bn3_out = self.bottleneck3(bn2_out)

                bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
                bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))

                interpolated = F.interpolate(bn5_out, mode='bilinear', scale_factor=2)
                return self.final(interpolated)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)


ENet = ENet_2_5d