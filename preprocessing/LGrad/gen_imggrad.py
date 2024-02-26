
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from glob import glob
import cv2
import sys
sys.path.append('.')
import os
import PIL.Image
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from preprocessing_model.LGrad_models import build_model
from PIL import ImageFile
from io import BytesIO
from scipy.ndimage import gaussian_filter
ImageFile.LOAD_TRUNCATED_IMAGES = True



processimg = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
#            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
        ])

def pil_jpg_eval(img, compress_val):
    out = BytesIO()
    img.save(out, format='jpeg', quality=compress_val)
    img = PIL.Image.open(out)
    img = np.array(img)
    img = PIL.Image.fromarray(img)
    out.close()
    return img

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    
    
def read_batchimg(imgpath_list, noise_type):
    img_list = []
    for imgpath in imgpath_list:
        img = PIL.Image.open(imgpath).convert('RGB')
        height,weight=img.height, img.width


        #这里是新加入的对图像的预处理
        if noise_type == 'jpg':
            # print('jpg')
            # quit()
            img_processed = pil_jpg_eval(img,95)
        elif noise_type == 'resize':
            # print('resize')
            img_processed = torchvision.transforms.Resize((int(height/2),int(weight/2)))(img) #可以改成缩放多少倍，因为图像质量各不相同


        elif noise_type == 'blur':
            # print('blur')
            img = np.array(img)
            gaussian_blur(img, 1)
            img_processed =  PIL.Image.fromarray(img)
        else:
            # print('none')
            img_processed = img
        # 对图像进行处理的代码
        # 注意通道数，在pytorch中，第一列是batchsize，第二列是通道数，第三列之后是图像
        
        # img = Image.open(filepath).convert('RGB')
        # img = pil_jpg_eval(img,95)
        # img_processed = np.array(img_processed).astype(np.float32)
        # img_processed = img_processed/255.
        img_list.append(torch.unsqueeze(processimg(img_processed),0))
        # img_list.append( torch.unsqueeze(processimg(PIL.Image.open(imgpath).convert('RGB')),0)  )
    return torch.cat(img_list,0)
    
def normlize_np(img):
    img -= img.min()
    if img.max()!=0: img /= img.max()
    return img * 255.                

def get_imglist(path):
    ext = [".jpg",".bmp",".png",".jpeg",".webp",".JPEG",'PNG']   # Add image formats here
    files = []
    [files.extend(glob(os.path.join(path, f'*{e}'))) for e in ext]
    return sorted(files)

def generate_images():
    imgdir = sys.argv[1]
    outdir = sys.argv[2]
    modelpath = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv)>4 else 1
    noise_type=sys.argv[5]
    print(noise_type)
    
    print(f'Transform {imgdir} to {outdir}')
    os.makedirs(outdir, exist_ok=True)

    # 将下载的预训练stylegan的分类器进行转换，变成pytorch版本
    model = build_model(gan_type='stylegan',
        module='discriminator',
        resolution=256,
        label_size=0,
        # minibatch_std_group_size = 1,
        image_channels=3)
    model.load_state_dict(torch.load(modelpath), strict=True)

    model.cuda()
    model.eval()

    imgnames_list = get_imglist(imgdir)
    if len(imgnames_list) == 0: exit()
    num_items = len(imgnames_list)
    print(f'From {imgdir} read {num_items} Img')
    minibatch_size = int(batch_size)
    numnow = 0
    for mb_begin in range(0, num_items, minibatch_size):
        imgname_list = imgnames_list[mb_begin: min(mb_begin+minibatch_size,num_items)]
        
        # idx=0
        # print(f'Gen grad to {os.path.join(outdir, imgname_list[idx].split("/")[-1])}')
        # quit()
        
        imgs_np      = read_batchimg(imgname_list,noise_type)
  
        tmpminibatch = len(imgname_list)
        img_cuda = imgs_np.cuda().to(torch.float32)
        img_cuda.requires_grad = True
        pre = model(img_cuda)
        model.zero_grad()
        grad = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
        
        for idx in range(tmpminibatch):
            numnow += 1
            img = normlize_np(grad[idx].permute(1,2,0).cpu().detach().numpy())
            print(f'Gen grad to {os.path.join(outdir, imgname_list[idx].split("/")[-1])}, bs:{minibatch_size} {numnow}/{num_items}',end='\r')

            cv2.imwrite(os.path.join(outdir, imgname_list[idx].split('/')[-1].split('.')[0]+'.png'),img[...,::-1])
    print()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
