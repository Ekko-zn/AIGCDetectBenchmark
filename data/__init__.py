import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder
from .datasets import read_data_new
from torch.utils.data import Dataset

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler



# PSM
def patch_collate_test(batch):
    input_img=[item[0] for item in batch]
    cropped_img=torch.stack([item[1] for item in batch], dim=0)
    target=torch.tensor([item[2] for item in batch])
    scale=torch.stack([item[3] for item in batch], dim=0)
    filename=[item[4] for item in batch]
    return [input_img, cropped_img, target, scale, filename]



# CNNSpot
def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader






    
def create_dataloader_new(opt):
    shuffle = True if opt.isTrain else False
    dataset = read_data_new(opt)
    if opt.detect_method=='Fusing':
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              collate_fn=patch_collate_test,
                                              num_workers=int(0))
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=int(0))
    return data_loader

