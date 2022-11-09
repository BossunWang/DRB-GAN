import torch
from torch import nn

from DynamicConv import AdaDynamicConv
from Decoder import Decoder
from basic_layer import Conv2dBlock, AdaILN


class ContentEncoder(nn.Module):
    def __init__(self, out_channels):
        super(ContentEncoder, self).__init__()

        cfg = [32, out_channels]
        layers = []
        in_channels = 3
        for v in cfg:
            layers += [Conv2dBlock(in_channels, v, kernel_size=3, stride=2, padding=1
                                   , pad_type="reflect", norm="ln", activation="relu")]
            in_channels = v

        self.feature_layer = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.feature_layer(x)
        return feature


class DynamicResBlock(nn.Module):
    def __init__(self, in_ch, gamma_dim, beta_dim, omega_dim, out_ch):
        super(DynamicResBlock, self).__init__()

        assert gamma_dim == beta_dim
        assert gamma_dim == omega_dim
        self.use_res_connect = in_ch == out_ch
        bottleneck = gamma_dim
        self.padd_layer = nn.ReflectionPad2d(1)
        self.conv_layer1 = nn.Conv2d(in_ch, bottleneck, kernel_size=3, stride=1, padding=0, bias=False)
        self.normal_layer1 = AdaILN(bottleneck)
        self.relu_layer = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(bottleneck, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.normal_layer2 = AdaILN(bottleneck)

    def forward(self, input, style_gamma_code, style_beta_code, style_omega_code):
        out = self.padd_layer(input)
        out = self.conv_layer1(out)
        out = self.normal_layer1(out, style_gamma_code, style_beta_code, style_omega_code)
        out = self.relu_layer(out)
        out = self.padd_layer(out)
        out = self.conv_layer2(out)
        out = self.normal_layer2(out, style_gamma_code, style_beta_code, style_omega_code)
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

        self.decoder = Decoder(enc_out_ch)

    def forward(self, x, style_gamma_code, style_beta_code, style_omega_code):
        content_feature = self.content_encoder(x)

        res_feature = content_feature
        for i, res_layer in enumerate(self.res_layers):
            res_feature = res_layer(res_feature, style_gamma_code, style_beta_code, style_omega_code)

        output = self.decoder(res_feature)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    encoder_out_ch = 64
    gamma_dim = 64
    beta_dim = 64
    omega_dim = 64
    db_number = 8
    ws = 64
    style_transfer_net \
        = StyleTransferNetwork(encoder_out_ch, gamma_dim, beta_dim, omega_dim, db_number, ws).to(device)

    content_input = torch.rand(1, 3, 256, 256).to(device)
    style_gamma = torch.rand(1, gamma_dim).to(device)
    style_beta = torch.rand(1, beta_dim).to(device)
    style_omega = torch.rand(1, omega_dim).to(device)
    output = style_transfer_net(content_input, style_gamma, style_beta, style_omega)
    print(output.size())
