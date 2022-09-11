import torch
from torch import nn
import numpy as np

from basic_layer import ConvSpectralNorm, ConvNormLReLU


class Discriminator(nn.Module):
    def __init__(self, M):
        super(Discriminator, self).__init__()

        feature_extraction_cfg = [64, 128]
        d_cfg = [256, 512]
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

    def forward(self, x, collection_s, shuffle=False):
        feature_list = torch.tensor([], dtype=torch.float, device=x.device)
        x_feature = self.feature_layer(x)
        N, C, H, W = x_feature.size()
        feature_list = torch.cat([feature_list, x_feature])

        collection_s = collection_s.view(-1, x.size(1), x.size(2), x.size(3))
        for s in collection_s:
            s_feature = self.feature_layer(s.unsqueeze(0))
            feature_list = torch.cat([feature_list, s_feature])

        if shuffle:
            random_index = torch.randperm(len(feature_list)).long().to(x.device)
            feature = feature_list[random_index].view(N, -1, H, W)
        else:
            feature = feature_list.view(N, -1, H, W)

        out_prep = self.D(feature)

        return out_prep


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 2
    discriminator = Discriminator(M + 1).to(device)
    content_input = torch.rand(1, 3, 256, 256).to(device)
    style_collection_inputs = torch.rand(1, 3 * M, 256, 256).to(device)

    out_prep = discriminator(content_input, style_collection_inputs, shuffle=True)
    print("out_prep:", out_prep)