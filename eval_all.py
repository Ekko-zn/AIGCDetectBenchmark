'''
python eval_all.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,PSM,DCTAnalysis,LGrad,LNP,DIRE}  --noise_type {blur,jpg,resize}
'''

import os
import csv
import torch

from validate import validate,validate_single
from networks.resnet import resnet50
from networks.Patch5Model import Patch5Model
from base_options import BaseOptions
from eval_config import *
import networks.resnet_gram as ResnetGram
import torchvision
from PIL import ImageFile
from util import create_argparser
import numpy as np
import random
from networks import univfd_models

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义一个函数，根据 model_type 返回相应的模型
def get_model(opt):
    if opt.detect_method in ["CNNSpot","LNP","LGrad","DIRE"]:
        return resnet50(num_classes=1)
    elif opt.detect_method == "DCTAnalysis":
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048,1)
        model.cuda()
        return model
    elif opt.detect_method == "PSM":
        return Patch5Model()
    elif opt.detect_method == "Gram":
        return ResnetGram.resnet18(num_classes=1)
    elif opt.detect_method == "UnivFD":
        opt.arch = 'CLIP:ViT-L/14'
        return univfd_models.get_univfd_model(opt.arch)
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)




# 固定随机种子
set_random_seed()
# Running tests


opt = BaseOptions().parse(print_options=True) #获取参数类



model_name = os.path.basename(opt.model_path).replace('.pth', '')
results_dir=f"./results/{opt.detect_method}"
mkdir(results_dir)

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)

    # model = resnet50(num_classes=1)
    model = get_model(opt)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    try:
        if opt.detect_method in ["DCTAnalysis","Gram"]:
            model.load_state_dict(state_dict['netC'],strict=True)
        elif opt.detect_method == "UnivFD":
            model.fc.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['model'],strict=True)
    except:
        print("[ERROR] model.load_state_dict() error")
    model.cuda()
    model.eval()


    opt.process_device=torch.device("cpu")
    acc, ap, _, _, _, _ = validate(model, opt,val)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))


# 结果文件
csv_name = results_dir + '/{}_{}.csv'.format(opt.detect_method,opt.noise_type)
with open(csv_name, 'a+') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
