"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import sys
sys.path.append('.')
# print(sys.path)
import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
from networks.denoising_rgb import DenoiseNet
from preprocessing_model.LNP_model.dataloaders.data_rgb import get_validation_data
import util
import cv2
from skimage import img_as_ubyte





parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/data',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='the LNP pic dic',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./weights/preprocessing/sidd_rgb.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--noise_type', default=None,
    type=str,help='e.g. jpg, blur, resize')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

util.mkdir(args.result_dir)
count = 0

def get_image_paths(folder_path):
    image_paths = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.PNG','.JPEG')):
            image_paths.append(file_path)
    
    return image_paths


keys=['stylegan2/car','stylegan2/cat','stylegan2/church','stylegan2/horse','whichfaceisreal'] # replace with your own dataset
for key in keys:
    image_paths=[]
    for sub in ['0_real','1_fake']:
        image_paths.extend(get_image_paths(f'{args.input_dir}/{key}/{sub}'))
        
    test_dataset = get_validation_data(image_paths,args.noise_type)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print(key+':'+str(len(test_dataset)))
    print('storing into: '+args.result_dir)
    model_restoration = DenoiseNet()
    util.load_checkpoint(model_restoration,args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration=nn.DataParallel(model_restoration)
    model_restoration.eval()
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            try:
                rgb_restored = model_restoration(rgb_noisy)
            except:
                print(filenames)
                continue
            rgb_restored = torch.clamp(rgb_restored,0,1)

            rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()


            for batch in range(len(rgb_noisy)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                imgsavepath = filenames[batch].replace(args.input_dir,args.result_dir)
           
                imgsavepath = imgsavepath.replace('jpg','png').replace('JPEG','png').replace('jpeg','png')
                try:
                    rootpath = imgsavepath.split('/')
                    rootpath.pop(-1)
                    rootpath = '/'.join(rootpath)
                    os.makedirs(rootpath)
                    print('root_path:'+rootpath)

                    cv2.imwrite(imgsavepath, denoised_img * 255)
                except:
                    cv2.imwrite(imgsavepath, denoised_img * 255)

                    

