import torch
from torch import nn
import torch.nn.functional as F


class SW_LIN(nn.Module):
    def __init__(self, normalized_shape, ws, eps=1e-6):
        super(SW_LIN, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.rho = nn.Parameter(torch.ones(1) * 0.5)
        self.ws = ws

    def forward(self, x):
        # calculate center region
        size = x.size()
        sw1 = (x.size(3) - self.ws) // 2
        sw2 = x.size(3) - (x.size(3) - self.ws) // 2
        sh1 = (x.size(2) - self.ws) // 2
        sh2 = x.size(2) - (x.size(2) - self.ws) // 2
        x_loc = x[:, :, sh1: sh2, sw1: sw2]
        # Layer Normalization
        ul = x_loc.reshape(x_loc.size(0), -1).mean(-1, keepdims=True)
        sl = (x_loc.reshape(x_loc.size(0), -1) - ul).pow(2).mean(-1, keepdims=True)
        x_ln = (x - ul.expand(size)) / torch.sqrt(sl.expand(size) + self.eps)

        # Instance Normalization
        ui = x_loc.reshape(x_loc.size(0), x_loc.size(1), -1).mean(-1, keepdims=True)
        si = (x_loc.reshape(x_loc.size(0), x_loc.size(1), -1) - ui).pow(2).mean(-1, keepdims=True)
        x_in = (x - ui.view(x.size(0), x.size(1), 1, 1).expand(size)) \
               / torch.sqrt(si.view(x.size(0), x.size(1), 1, 1).expand(size) + self.eps)

        x = self.weight[:, None, None] * (self.rho * x_in + (1. - self.rho) * x_ln) + self.bias[:, None, None]

        return x


class SW_LIN_Decoder(nn.Module):
    def __init__(self, in_channels, ws):
        super(SW_LIN_Decoder, self).__init__()

        cfg = [in_channels // 2, in_channels // 4]
        ws_group = [ws, ws * 2]
        layers = []
        for i, (v, ws) in enumerate(zip(cfg, ws_group)):
            if i == len(cfg) - 1:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=7, stride=1, padding=3)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=2)
            layers += [nn.Sequential(conv2d, SW_LIN(v, ws), nn.ReLU(inplace=True))]
            in_channels = v

        self.decode_layer = nn.ModuleList(layers)
        self.out_layer = nn.Sequential(
            nn.Conv2d(cfg[-1], 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x, align_corners=True):
        output = x
        for l in self.decode_layer:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=align_corners)
            output = l(output)

        output = self.out_layer(output)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_shape = 128
    ws = 64
    sw_in = SW_LIN(normalized_shape, ws).to(device)

    content_feature = torch.rand(1, normalized_shape, 127, 127).to(device)
    content_normal = sw_in(content_feature)
    print(content_normal.size())

    in_channels = 128
    decoder = SW_LIN_Decoder(in_channels, ws).to(device)
    output = decoder(content_feature)
    print(output.size())