import torch
from torch import nn

from ADIN import ADIN_Dynamic
from DynamicConv import DynamicConv
from SW_LIN_Decoder import SW_LIN_Decoder
from basic_layer import ConvNormLReLU


class ContentEncoder(nn.Module):
    def __init__(self, out_channels):
        super(ContentEncoder, self).__init__()

        cfg = [64, out_channels]
        layers = []
        in_channels = 3
        for v in cfg:
            layers += [ConvNormLReLU(in_channels, v, kernel_size=3, stride=2)]
            in_channels = v

        self.feature_layer = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.feature_layer(x)
        return feature


class DynamicResBlock(nn.Module):
    def __init__(self, in_ch, gamma_dim, beta_dim, omega_dim, out_ch):
        super(DynamicResBlock, self).__init__()

        assert gamma_dim == beta_dim
        self.use_res_connect = in_ch == out_ch
        bottleneck = gamma_dim
        self.conv_layer = nn.Conv2d(in_ch, bottleneck, kernel_size=3, stride=1, padding=1, bias=False)
        self.adin_layer = ADIN_Dynamic()
        self.relu_layer = nn.ReLU(inplace=True)
        self.dy_conv_layer = DynamicConv(in_planes=bottleneck, out_planes=out_ch
                                         , kernel_size=3, stride=1, padding=1, bias=False, temprature=1, K=omega_dim)
        self.in_layer = nn.InstanceNorm2d(out_ch)

    def forward(self, input, style_gamma_code, style_beta_code, style_omega_code):
        out = self.conv_layer(input)
        out = self.adin_layer(out, style_gamma_code, style_beta_code)
        out = self.relu_layer(out)
        out = self.dy_conv_layer(out, style_omega_code)
        out = self.in_layer(out)
        if self.use_res_connect:
            out = input + out
        return out


class StyleTransferNetwork(nn.Module):
    def __init__(self, enc_out_ch, gamma_dim, beta_dim, omega_dim, db_number, ws):
        super(StyleTransferNetwork, self).__init__()

        self.content_encoder = ContentEncoder(enc_out_ch)

        db_layers = []
        for _ in range(db_number):
            db_layers += [DynamicResBlock(enc_out_ch, gamma_dim, beta_dim, omega_dim, enc_out_ch)]

        self.res_layers = nn.ModuleList(
            DynamicResBlock(enc_out_ch, gamma_dim, beta_dim, omega_dim, enc_out_ch) for _ in range(db_number))

        self.decoder = SW_LIN_Decoder(enc_out_ch, ws)

    def forward(self, x, style_gamma_code, style_beta_code, style_omega_code):
        content_feature = self.content_encoder(x)
        # print(content_feature.size())

        res_feature = content_feature
        for i, res_layer in enumerate(self.res_layers):
            res_feature = res_layer(res_feature, style_gamma_code, style_beta_code, style_omega_code)

        output = self.decoder(res_feature)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    encoder_out_ch = 128
    gamma_dim = 256
    beta_dim = 256
    omega_dim = 4
    db_number = 4
    ws = 64
    style_transfer_net \
        = StyleTransferNetwork(encoder_out_ch, gamma_dim, beta_dim, omega_dim, db_number, ws).to(device)

    content_input = torch.rand(1, 3, 256, 256).to(device)
    style_gamma = torch.rand(1, gamma_dim, 1, 1).to(device)
    style_beta = torch.rand(1, beta_dim, 1, 1).to(device)
    style_omega = torch.rand(1, omega_dim).to(device)
    output = style_transfer_net(content_input, style_gamma, style_beta, style_omega)
    print(output.size())
