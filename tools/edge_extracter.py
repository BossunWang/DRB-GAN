import torch
import torch.nn.functional as F


def edge_map(img, enhance=True):

    '''
    :param img: -1 ~ 1
    :param enhance:
    :return:
    '''

    def high_pass_filter(img, d, n):
        '''
        :param img: 0 ～ 1
        :return:
        '''
        return 1 - 1 / (1 + (img / d) ** n)

    # 1, 3, H, W  -1 ~ 1
    v_kernel = torch.tensor(
        [[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
         [[0.0, -2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
         [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).unsqueeze(0).float().to(img.device)

    h_kernel = torch.tensor(
        [[[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]).unsqueeze(0).float().to(img.device)

    v_center_kernel = torch.tensor(
        [[[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],
         [[0.0, -2.0, 0.0], [0.0, 4.0, 0.0], [0.0, -2.0, 0.0]],
         [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]]]).unsqueeze(0).float().to(img.device)

    h_center_kernel = torch.tensor(
        [[[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-2.0, 4.0, -2.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]]]).unsqueeze(0).float().to(img.device)

    kernel = torch.cat([v_kernel, h_kernel, v_center_kernel, h_center_kernel], dim=0)

    p2d = (1, 1, 1, 1)
    img_pad = F.pad(img, p2d, "reflect")
    edge = F.conv2d(img_pad, kernel, bias=None, stride=1, padding=0).sum(dim=1, keepdim=True)

    # normalize to 0 ~ 1
    edge = (edge - torch.min(edge)) / (torch.max(edge) -torch.min(edge))

    if enhance:
        edge = high_pass_filter(edge, d=0.2, n=2)

    # show img
    # import cv2
    # import numpy as np
    # e = edge.data.cpu().numpy().squeeze(0).squeeze(0)
    # e = np.asarray(e * 255, dtype=np.uint8)
    # cv2.imshow("edge", e)
    # cv2.waitKey(0)

    return edge





