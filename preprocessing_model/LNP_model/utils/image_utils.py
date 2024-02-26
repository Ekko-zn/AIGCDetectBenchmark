import torch
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pickle
# import lycon
import cv2
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from io import BytesIO
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def pil_jpg_eval(img, compress_val):
    out = BytesIO()
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    img = Image.fromarray(img)
    out.close()
    return img

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


#在load_img中增加了noise_type，选择使用的噪声，如jpg压缩、高斯模糊等等
def load_img(filepath, noise_type):
    '''
    noise_type: 选择使用的噪声, 如jpg压缩、高斯模糊等等
    '''
    # img = cv2.imread(filepath)
    # img = img.astype(np.float32)
    # img = img/255.
  
    img = Image.open(filepath).convert('RGB')
    if noise_type == 'jpg':
        img_processed = pil_jpg_eval(img,95)
    elif noise_type == 'resize':
        width, height = img.size
        # print('before resize: '+str(width)+str(height))
        img_processed = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 
        #可以改成缩放多少倍，因为图像质量各不相同
        # width, height = img_processed.size
        # print('after resize: '+str(width)+str(height))
    elif noise_type == 'blur':
        # print('blur')
        img = np.array(img)
        gaussian_blur(img, 1)
        img_processed = Image.fromarray(img)
    else:
        img_processed = img
    # 对图像进行处理的代码
    # 注意通道数，在pytorch中，第一列是batchsize，第二列是通道数，第三列之后是图像
    
    # img = Image.open(filepath).convert('RGB')
    # img = pil_jpg_eval(img,95)
    img_processed = np.array(img_processed).astype(np.float32)
    img_processed = img_processed/255.
    return img_processed

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        psnr = compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        if np.isinf(psnr):
            continue
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)


def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = []
    for i in range(Img.shape[0]):
        ssim = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], gaussian_weights=True, use_sample_covariance=False, multichannel =True)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM)


def unpack_raw(im):
    bs,chan,h,w = im.shape 
    H, W = h*2, w*2
    img2 = torch.zeros((bs,H,W))
    img2[:,0:H:2,0:W:2]=im[:,0,:,:]
    img2[:,0:H:2,1:W:2]=im[:,1,:,:]
    img2[:,1:H:2,0:W:2]=im[:,2,:,:]
    img2[:,1:H:2,1:W:2]=im[:,3,:,:]
    img2 = img2.unsqueeze(1)
    return img2

def pack_raw(im):
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:],
                       im[1:H:2,1:W:2,:]), axis=2)
    return out

def pack_raw_torch(im):
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = torch.cat((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:],
                       im[1:H:2,1:W:2,:]), dim=2)
    return out
