import torch
from torch import nn


class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        cfg = [64, 64, 128, 256, 512]
        layers = []
        in_channels = 3
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2)
            layers += [conv2d, nn.GroupNorm(num_groups=1, num_channels=v, affine=True), nn.ReLU(inplace=True)]
            in_channels = v

        self.feature_layer = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.feature_layer(x)
        return feature


class AuxiliaryClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def get_classifier_weights(self):
        return self.classifier.weight

    def forward(self, x):
        output = self.classifier(x)
        return output


class Style_MLP(nn.Module):
    def __init__(self, feature_dim, gamma_dim, beta_dim, omega_dim):
        super(Style_MLP, self).__init__()
        self.gamma_dim = gamma_dim  # ADIN mean shape: (N, C,)
        self.beta_dim = beta_dim  # ADIN std shape: (N, C,)
        self.omega_dim = omega_dim  # DConv weights shape: (N, K)

        self.linear = nn.Linear(feature_dim, gamma_dim + beta_dim + omega_dim)

    def forward(self, x):
        output = self.linear(x)
        return output[:, :self.gamma_dim] \
            , output[:, self.gamma_dim: self.gamma_dim + self.beta_dim] \
            , output[:, self.gamma_dim + self.beta_dim: self.gamma_dim + self.beta_dim + self.omega_dim]


class StyleEncodingNetwork(nn.Module):
    def __init__(self, feature_dim, num_classes, VGG_model, gamma_dim, beta_dim, omega_dim):
        super(StyleEncodingNetwork, self).__init__()

        self.VGG = VGG_model
        self.style_encoder = StyleEncoder()
        self.ac_classifier = AuxiliaryClassifier(feature_dim, num_classes)
        self.H = nn.ModuleList(
            Style_MLP(feature_dim, gamma_dim, beta_dim, omega_dim) for _ in range(num_classes))

    def forward(self, x):
        style_vgg_features = self.VGG(x)
        style_vgg_feature = style_vgg_features.relu5_1
        # print(style_vgg_feature.size())
        style_feature = self.style_encoder(x)

        style_vgg_feature = style_vgg_feature.view(style_vgg_feature.size(0), -1)
        style_feature = style_feature.view(style_feature.size(0), -1)

        style_mixed_feature = torch.cat([style_vgg_feature, style_feature], -1)
        # print(style_mixed_feature.size())

        style_prob = self.ac_classifier(style_mixed_feature)

        # style_mixed_feature multiply by classifier weights
        ac_weights = self.ac_classifier.get_classifier_weights().unsqueeze(0)
        style_mixed_feature_expand = style_mixed_feature.unsqueeze(1).expand(-1, ac_weights.size(1), -1)
        style_mixed_weights_feature = torch.mul(ac_weights, style_mixed_feature_expand)
        # print(style_mixed_weights_feature.size())

        style_gamma_code, style_beta_code, style_omega_code = [], [], []
        for i, h in enumerate(self.H):
            style_gamma, style_beta, style_omega = h(style_mixed_weights_feature[:, i])
            style_gamma_code.append(style_gamma)
            style_beta_code.append(style_beta)
            style_omega_code.append(style_omega)

        style_gamma_code = torch.cat(style_gamma_code).unsqueeze(-1).unsqueeze(-1)
        style_beta_code = torch.cat(style_beta_code).unsqueeze(-1).unsqueeze(-1)
        style_omega_code = torch.cat(style_omega_code)

        return style_prob, style_gamma_code, style_beta_code, style_omega_code


if __name__ == '__main__':
    from vgg_nets import Vgg19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VGG = Vgg19().to(device)
    VGG.eval()

    feature_dim = 156160
    num_classes = 4
    gamma_dim = 256
    beta_dim = 256
    omega_dim = 4
    style_encoding_net \
        = StyleEncodingNetwork(feature_dim, num_classes, VGG, gamma_dim, beta_dim, omega_dim).to(device)

    style_input = torch.rand(1, 3, 256, 256).to(device)
    style_prob, style_gamma, style_beta, style_omega = style_encoding_net(style_input)
    print("style_prob:", style_prob.size())
    print("style_gamma:", style_gamma.size())
    print("style_beta:", style_beta.size())
    print("style_omega:", style_omega.size())
