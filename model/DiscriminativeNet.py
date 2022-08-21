import torch
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np


class ConvSpectralNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(ConvSpectralNorm, self).__init__()

        layers = [spectral_norm(nn.Conv2d(in_ch
                                          , out_ch
                                          , kernel_size=kernel_size
                                          , stride=stride
                                          , padding=0
                                          , groups=groups
                                          , bias=bias))]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        return out


class Discriminator(nn.Module):
    def __init__(self, M):
        super(Discriminator, self).__init__()

        feature_extraction_cfg = [64, 64]
        d_cfg = [128, 256]
        feature_layers = []
        d_layers = []
        in_channels = 3
        for v in feature_extraction_cfg:
            feature_layers += [ConvSpectralNorm(in_channels, v, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
            in_channels = v

        in_channels *= M
        for v in d_cfg:
            d_layers += [ConvSpectralNorm(in_channels, v, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
            in_channels = v

        d_layers += [ConvSpectralNorm(in_channels, 1, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.feature_layer = nn.Sequential(*feature_layers)
        self.D = nn.Sequential(*d_layers)

    def forward(self, x, s_list, shuffle=False):
        feature_list = []
        x_feature = self.feature_layer(x)
        N, C, H, W = x_feature.size()
        feature_list.append(x_feature)

        for s in s_list:
            s_feature = self.feature_layer(s)
            feature_list.append(s_feature)

        if shuffle:
            random_index = np.random.permutation([i for i in range(len(feature_list))])
            feature = torch.cat(feature_list, dim=0)[random_index].view(N, -1, H, W)
        else:
            feature = torch.cat(feature_list, dim=0).view(N, -1, H, W)

        out_prep = self.D(feature)

        return out_prep


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 2
    discriminator = Discriminator(M + 1).to(device)
    content_input = torch.rand(1, 3, 512, 512).to(device)

    style_collection_inputs = [torch.rand(1, 3, 512, 512).to(device) for _ in range(M)]

    out_prep = discriminator(content_input, style_collection_inputs, shuffle=True)
    print("out_prep:", out_prep.size())