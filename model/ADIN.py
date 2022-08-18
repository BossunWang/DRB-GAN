import torch
from torch import nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ADIN_Dynamic(nn.Module):
    def __init__(self):
        super(ADIN_Dynamic, self).__init__()

    def forward(self, content_feat, style_mean, style_std):
        assert (content_feat.size()[:2] == style_mean.size()[:2])
        assert (content_feat.size()[:2] == style_std.size()[:2])
        size = content_feat.size()
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adin = ADIN_Dynamic().to(device)

    content_input = torch.rand(1, 256, 127, 127).to(device)
    style_mean = torch.rand(1, 256, 1, 1).to(device)
    style_std = torch.rand(1, 256, 1, 1).to(device)

    normalized_feat = adin(content_input, style_mean, style_std)
    print("normalized_feat:", normalized_feat.size())