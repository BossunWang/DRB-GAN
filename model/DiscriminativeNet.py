import torch
from torch import nn
import numpy as np

from basic_layer import ConvSpectralNorm, ConvNormLReLU, Conv2dBlock


class Discriminator(nn.Module):
    def __init__(self, M=2, num_f_layer=2, num_d_layer=3, num_scales=3
                 , input_dim=3, dim=64
                 , norm="none", activ="lrelu", pad_type="reflect"):
        super(Discriminator, self).__init__()

        self.M = M
        self.input_dim = input_dim
        self.dim = dim
        self.num_f_layer = num_f_layer
        self.num_d_layer = num_d_layer
        self.num_scales = num_scales
        self.norm = norm
        self.activ = activ
        self.pad_type = pad_type

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.feature_extraction = nn.ModuleList()
        self.D = nn.ModuleList()
        for _ in range(self.num_scales):
            feature_layers, d_layers = self._make_net()
            self.feature_extraction.append(feature_layers)
            self.D.append(d_layers)

    def _make_net(self):
        dim = self.dim

        # feature_extraction layer
        feature_layers = []
        feature_layers += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.num_f_layer - 1):
            feature_layers += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2

        # collection discriminator layer
        d_layers = []
        dim *= self.M
        for i in range(self.num_d_layer - 1):
            d_layers += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2

        d_layers += [nn.Conv2d(dim, 1, 1, 1, 0)]

        feature_layers = nn.Sequential(*feature_layers)
        d_layers = nn.Sequential(*d_layers)
        return feature_layers, d_layers

    def forward(self, x, collection_s, shuffle=False):
        collection_s = collection_s.view(-1, x.size(1), x.size(2), x.size(3))
        outputs = []
        for fe, d in zip(self.feature_extraction, self.D):

            feature_list = torch.tensor([], dtype=torch.float, device=x.device)
            x_feature = fe(x)
            N, C, H, W = x_feature.size()
            feature_list = torch.cat([feature_list, x_feature])

            for s in collection_s:
                s_feature = fe(s.unsqueeze(0))
                feature_list = torch.cat([feature_list, s_feature])

            if shuffle:
                random_index = torch.randperm(len(feature_list)).long().to(x.device)
                feature = feature_list[random_index].view(N, -1, H, W)
            else:
                feature = feature_list.view(N, -1, H, W)

            out_logit = d(feature)
            outputs.append(out_logit)
            x = self.downsample(x)
            collection_s = self.downsample(collection_s)

        return outputs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 2
    discriminator = Discriminator(M + 1).to(device)
    content_input = torch.rand(1, 3, 256, 256).to(device)
    style_collection_inputs = torch.rand(1, 3 * M, 256, 256).to(device)

    output_logits = discriminator(content_input, style_collection_inputs, shuffle=False)

    for output_logit in output_logits:
        print("output_logit:", output_logit.size())