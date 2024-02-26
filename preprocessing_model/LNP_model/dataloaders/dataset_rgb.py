import numpy as np
import os
from torch.utils.data import Dataset
import torch
from preprocessing_model.LNP_model.utils.image_utils import is_png_file, load_img
from preprocessing_model.LNP_model.utils.GaussianBlur import get_gaussian_kernel

import torch.nn.functional as F


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, paths, noise_type=None):
        super(DataLoaderVal, self).__init__()
        # self.paths = [path.replace('/share','') for path in paths] #替换掉/hotdata/CNNSpot_test/paths中保存路径中存在的/share
        self.paths=paths
        self.noise_type=noise_type
       


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        

        noisy_filename = self.paths[index]
        

        # 这里调用的loda_img中对图像进行了一个处理
        noisy = torch.from_numpy(np.float32(load_img(noisy_filename, self.noise_type)))

        

        noisy = noisy.permute(2, 0, 1)
        return noisy, noisy_filename


##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, '1_fake')))

        self.noisy_filenames = [os.path.join(rgb_dir, '1_fake', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2, 0, 1)

        return noisy, noisy_filename


##################################################################################################

MAX_SIZE = 512


def divisible_by(img, factor=16):
    h, w, _ = img.shape
    img = img[:int(np.floor(h / factor) * factor), :int(np.floor(w / factor) * factor), :]
    return img


class DataLoader_NoisyData(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoader_NoisyData, self).__init__()

        rgb_files = sorted(os.listdir(rgb_dir))

        self.target_filenames = [os.path.join(rgb_dir, x) for x in rgb_files]
        self.tar_size = len(self.target_filenames)  # get the size of target
        self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        target = np.float32(load_img(self.target_filenames[tar_index]))

        target = divisible_by(target, 16)

        tar_filename = os.path.split(self.target_filenames[tar_index])[-1]

        target = torch.Tensor(target)
        target = target.permute(2, 0, 1)

        target = F.pad(target.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        target = self.blur(target).squeeze(0)

        padh = (MAX_SIZE - target.shape[1]) // 2
        padw = (MAX_SIZE - target.shape[2]) // 2
        target = F.pad(target.unsqueeze(0), (padw, padw, padh, padh), mode='constant').squeeze(0)

        return target, tar_filename, padh, padw
