import os
import torch
from collections import OrderedDict
import argparse

import torch.nn as nn
import numpy as np
import random
import torchvision

import networks.univfd_models as univfd_models
import networks.resnet_gram as ResnetGram
from networks.Patch5Model import Patch5Model
from networks.resnet import resnet50

from preprocessing_model.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        

def create_argparser():
    defaults = dict(
        images_dir="./test/stargan",
        recons_dir="./test_dire/recons",
        dire_dir="./test_dire/dire",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        model_path="./weights/prepocessing/lsun_bedroom.pt",
        real_step=0,  #test
        continue_reverse=False,
        has_subfolder=False,
        has_subclasses=False,
        timestep_respacing="ddim20"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_model(opt):
    if opt.detect_method in ["CNNSpot","LNP","LGrad","DIRE"]:
        if opt.isTrain:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(model.fc.weight.data, 0.0, opt.init_gain)
            return model
        else:
            return resnet50(num_classes=1)
    elif opt.detect_method == "FreDect":
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048,1)
        return model
    elif opt.detect_method == "Fusing":
        return Patch5Model()
    elif opt.detect_method == "Gram":
        return ResnetGram.resnet18(num_classes=1)
    elif opt.detect_method == "UnivFD":
        opt.arch = 'CLIP:ViT-L/14'
        model = univfd_models.get_univfd_model(opt.arch)
        if opt.isTrain:
            torch.nn.init.normal_(model.fc.weight.data, 0.0, opt.init_gain) 
        return model

    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")