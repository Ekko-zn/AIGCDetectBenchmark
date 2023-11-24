import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import torchvision
import os
import copy
import torch
from scipy import fftpack
import imageio
from skimage.transform import resize
from .process import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize) # 随机剪裁，默认224
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img) # 不处理
    else:
        crop_func = transforms.CenterCrop(opt.cropSize) # 中心裁剪

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    # dset = datasets.ImageFolder(
    #         root,
    #         transforms.Compose([
    #             rz_func,
    #             # 加入了变成灰度图像的操作
    #             transforms.Grayscale(num_output_channels=3),
    #             transforms.Lambda(lambda img: data_augment(img, opt)),
    #             crop_func,
    #             flip_func,
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485], std=[0.229]),
    #         ]))
    # return dset
    # 原始的预处理过程
    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    width, height = img.size
    img = np.array(img)

    if random() < opt.blur_prob:

        sig = sample_continuous(opt.blur_sig)
        # print('blur:'+str(sig))
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    # resize
    
    # print('before resize: '+str(width)+str(height))
    # img_processed = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)




def cv2_jpg_gray(img, compress_val):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 0)
    return decimg

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 
    # return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
    return img


def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        if opt.detect_method=='PSM':
            height, width = img.shape[0], img.shape[1]
            # print('width, heig/ht:'+str(width)+','+str(height))
            img = resize(img, (int(height/2), int(width/2)))
        else:
            height, width = img.height, img.width
            img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]
    if opt.noise_type=='blur':
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    return Image.fromarray(img)


def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '0_real' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths

def process_img(img,opt,imgname,target):
    if opt.detect_method in ['CNNSpot','Gram']:
        img = processing(img,opt)
    elif opt.detect_method == 'DCTAnalysis':
        img = processing_DCT(img,opt)
    elif opt.detect_method == 'PSM':
        input_img, cropped_img, scale = processing_PSM(img,opt)
        return input_img, cropped_img, target, scale, imgname
    elif opt.detect_method == 'LGrad':
        opt.cropSize=256
        img = processing_LGrad(img, opt.gen_model, opt)
    elif opt.detect_method == 'LNP':
        img = processing_LNP(img, opt.model_restoration, opt, imgname)
    elif opt.detect_method == 'DIRE':
        img = processing_DIRE(img,opt,imgname)
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")


    return img, target
    



class read_data_new():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        # real_img_list=[]    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        # compute scaling
        height, width = img.height, img.width
        
        img = custom_augment(img, self.opt)
        
        
        
        
        if self.opt.detect_method in ['CNNSpot','Gram','Steg']:
            img = processing(img,self.opt)
        elif self.opt.detect_method == 'DCTAnalysis':
            img = processing_DCT(img,self.opt)
        elif self.opt.detect_method == 'PSM':
            input_img, cropped_img, scale = processing_PSM(img,self.opt)
            return input_img, cropped_img, target, scale, imgname
        elif self.opt.detect_method == 'LGrad':
            self.opt.cropSize=256
            img = processing_LGrad(img,self.opt.gen_model,self.opt)
        elif self.opt.detect_method == 'LNP':
            img = processing_LNP(img,self.opt.model_restoration,self.opt,imgname)
        elif self.opt.detect_method == 'DIRE':
            img = processing_DIRE(img,self.opt,imgname)
        elif self.opt.detect_method == 'UnivFD':
            img = processing_UnivFD(img,self.opt)
        else:
            raise ValueError(f"Unsupported model_type: {self.opt.detect_method}")


        return img, target

    def __len__(self):
        return len(self.label)
