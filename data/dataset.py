"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for dirPath, dirNames, fileNames in os.walk(dir):
        for f in fileNames:
            path = os.path.join(dirPath, f)
            if "jpg" in Path(path).suffix or "png" in Path(path).suffix:
                images.append(path)

    return images


class ImageDataset(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.train_list = []
        self.files = make_dataset(self.data_root)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def to_Tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_path = self.files[index]
        image = self.to_Tensor(image_path)
        return image


class ImageClassDataset(Dataset):
    def __init__(self, data_root, transform, sample_size=1):
        self.data_root = data_root
        self.label_dict = os.listdir(data_root)
        self.files = make_dataset(self.data_root)
        self.transform = transform
        self.sample_size = sample_size

    def __len__(self):
        return len(self.label_dict)

    def to_Tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_dir = self.label_dict[index]
        path, dirs, files = next(os.walk(os.path.join(self.data_root, image_dir)))
        file_index_list = np.random.permutation(len(files))[:self.sample_size]

        image_list = []
        label_list = []
        for fi in file_index_list:
            image_path = os.path.join(self.data_root, image_dir, files[fi])
            image = self.to_Tensor(image_path)
            image_list.append(image)
            label_list.append(index)
        return image_list, label_list


if __name__ == '__main__':
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 256
    crop_size = 256
    sample_isze = 3
    train_transform = [T.RandomHorizontalFlip()
                       , T.Resize(input_size, InterpolationMode.BICUBIC)
                       , T.RandomCrop(crop_size)
                       , T.ToTensor()
                       , T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    train_transform = T.Compose(train_transform)
    test_transform = [T.Resize(input_size, InterpolationMode.BICUBIC)
                      , T.ToTensor()
                      , T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    test_transform = T.Compose(test_transform)

    train_data_src = ImageDataset('../../photo2fourcollection-20210312T150938Z-001/photo2fourcollection/trainA'
                                  , train_transform)
    train_data_tgt = ImageClassDataset('../../photo2fourcollection-20210312T150938Z-001/photo2fourcollection/trainB'
                                       , train_transform
                                       , sample_isze)
    test_data_tgt = ImageDataset('../../photo2fourcollection-20210312T150938Z-001/photo2fourcollection/testA'
                                 , test_transform)

    batch_size = 1
    train_loader_src = torch.utils.data.DataLoader(train_data_src
                                                   , batch_size=batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    train_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                   , batch_size=batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    test_loader_src = torch.utils.data.DataLoader(test_data_tgt
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , drop_last=False)

    style_batch_iterator = iter(train_loader_tgt)

    mean_array = np.array((0.485, 0.456, 0.406)).reshape(1, 1, -1)
    std_array = np.array((0.229, 0.224, 0.225)).reshape(1, 1, -1)

    for content_image in tqdm(train_loader_src):
        content_image = content_image.to(device)
        content_image = content_image[0].cpu().numpy().transpose(1, 2, 0)

        content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        content_image = (content_image * std_array) + mean_array
        content_image = np.clip(content_image * 255.0, 0, 255)
        cv2.imwrite('sample_content.png', content_image)

        try:
            style_image_list, label_list = next(style_batch_iterator)
        except StopIteration:
            style_batch_iterator = iter(train_loader_tgt)
            style_image_list, label_list = next(style_batch_iterator)

        style_image = style_image_list[0][0].cpu().numpy().transpose(1, 2, 0)
        style_image = (style_image * std_array) + mean_array
        style_image = np.clip(style_image * 255.0, 0, 255)
        cv2.imwrite('sample_style.png', style_image)

