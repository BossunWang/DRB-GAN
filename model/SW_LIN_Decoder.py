import torch
from torch import nn
import torch.nn.functional as F
from basic_layer import SW_LIN


class SW_LIN_Decoder(nn.Module):
    def __init__(self, in_channels, ws):
        super(SW_LIN_Decoder, self).__init__()

        cfg = [in_channels // 2, in_channels // 4]
        ws_group = [ws, ws * 2]
        layers = []
        for i, (v, ws) in enumerate(zip(cfg, ws_group)):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=0)
            layers += [nn.Sequential(nn.ReflectionPad2d(1), conv2d, SW_LIN(v, ws), nn.ReLU(inplace=True))]
            in_channels = v

        self.decode_layer = nn.ModuleList(layers)
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cfg[-1], 3, kernel_size=3, stride=1, padding=0, bias=False),
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

    content_feature = torch.rand(1, normalized_shape, 128, 128).to(device)
    content_normal = sw_in(content_feature)
    print(content_normal.size())

    in_channels = 128
    decoder = SW_LIN_Decoder(in_channels, ws).to(device)
    output = decoder(content_feature)
    print(output.size())