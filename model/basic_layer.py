from torch import nn
from torch.nn.utils import spectral_norm


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=False):

        super(ConvNormLReLU, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


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