import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from random import random, choice
import copy
from scipy import fftpack
from skimage import img_as_ubyte
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.functional as TF

from util import load_checkpoint,create_argparser
from networks.denoising_rgb import DenoiseNet
from preprocessing_model.LGrad_models import build_model
from preprocessing_model.guided_diffusion import dist_util, logger
from preprocessing_model.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:

        sig = sample_continuous(opt.blur_sig)
        # print('blur:'+str(sig))
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)



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



def get_processing_model(opt):
    if opt.detect_method=='LGrad':
        gen_model = build_model(gan_type='stylegan',
            module='discriminator',
            resolution=256,
            label_size=0,
            # minibatch_std_group_size = 1,
            image_channels=3)
        gen_model.load_state_dict(torch.load(opt.LGrad_modelpath), strict=True)
        gen_model.to(opt.process_device)
        gen_model.eval()
        opt.gen_model=gen_model
        
        
    elif opt.detect_method == 'LNP':
        model_restoration = DenoiseNet()
        load_checkpoint(model_restoration,opt.LNP_modelpath)
        print("===>Testing using weights: ", opt.LNP_modelpath)
        # model_restoration=nn.DataParallel(model_restoration)
        model_restoration.to(opt.process_device)
        model_restoration.eval()
        opt.model_restoration=model_restoration
    
    elif opt.detect_method == 'DIRE':
        DIRE_args,_ = create_argparser().parse_known_args() # DIRE载入diffusion模型所需参数
        if opt.isTrain:
            DIRE_args.use_fp16=True
        opt.DIRE_args=DIRE_args
        print(DIRE_args)
        diffusion_model, diffusion = create_model_and_diffusion(**args_to_dict(DIRE_args, model_and_diffusion_defaults().keys()))
        print(opt.DIRE_modelpath)
        diffusion_model.load_state_dict(torch.load(opt.DIRE_modelpath, map_location="cuda"))
        
        diffusion_model.to(opt.process_device)
        logger.log("have created model and diffusion")
        if opt.DIRE_args.use_fp16:
            diffusion_model.convert_to_fp16()
        diffusion_model.eval()
        opt.diffusion_model=diffusion_model
        opt.diffusion=diffusion
        
    elif opt.detect_method =='FreDect':
        opt.dct_mean = torch.load('./weights/auxiliary/dct_mean').permute(1,2,0).numpy()
        opt.dct_var = torch.load('./weights/auxiliary/dct_var').permute(1,2,0).numpy()
        
    elif opt.detect_method in ['CNNSpot','Gram','Steg','Fusing',"UnivFD"]:
        opt=opt
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")
    return opt



def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    # width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    # img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
    # return img



def processing(img,opt):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.CropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.CropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    trans = transforms.Compose([
                rz_func,
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    return trans(img)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def processing_UnivFD(img,opt):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.CropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.CropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    trans = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt) if opt.isTrain else img),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN['clip'], std=STD['clip'] ),
                ])
    return trans(img)



def normlize_np(img):
    img -= img.min()
    if img.max()!=0: img /= img.max()
    return img * 255.   

processimg = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
#            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
        ])
def processing_LGrad(img,gen_model,opt):
    img_list = []
    img_list.append(torch.unsqueeze(processimg(img),0))
    img=torch.cat(img_list,0)
    img_cuda = img.to(torch.float32)
    img_cuda= img_cuda.to(opt.process_device)
    img_cuda.requires_grad = True
    pre = gen_model(img_cuda)
    gen_model.zero_grad()
    grads = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
    for idx,grad in enumerate(grads):
        img_grad = normlize_np(grad.permute(1,2,0).cpu().detach().numpy())
    retval, buffer = cv2.imencode(".png", img_grad)
    if retval:
        img = Image.open(BytesIO(buffer)).convert('RGB')
    else:
        print("保存到内存失败")
    img=processing(img,opt)
    return img







def dct2_wrapper(image, mean, var, log=True, epsilon=1e-12):
    """apply 2d-DCT to image of shape (H, W, C) uint8
    """
    # dct
    image = np.array(image)
    image = fftpack.dct(image, type=2, norm="ortho", axis=0)
    image = fftpack.dct(image, type=2, norm="ortho", axis=1)
    # log scale
    if log:
        image = np.abs(image)
        image += epsilon  # no zero in log
        image = np.log(image)
    # normalize
    image = (image - mean) / np.sqrt(var)
    return image



def processing_DCT(img,opt):
    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

    img = transforms.Resize(opt.loadSize)(img)
    img = transforms.CenterCrop(opt.CropSize)(img)
    cropped_img = torch.from_numpy(dct2_wrapper(img, opt.dct_mean, opt.dct_var)).permute(2,0,1).to(dtype=torch.float)
    return cropped_img


def processing_PSM(img,opt):
    height, width = img.height, img.width

    
    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

    img = transforms.Resize(opt.CropSize)(img)
    img = transforms.CenterCrop(opt.CropSize)(img)
    cropped_img = transforms.ToTensor()(img)
    cropped_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)


    scale = torch.tensor([height, width])

    return input_img, cropped_img, scale

def processing_LNP(img,model_restoration,opt,imgname):
    img_list = []
    img = np.array(img).astype(np.float32)
    img = img/255.
    img = torch.from_numpy(np.float32(img))
    img = img.permute(2, 0, 1)
    img_list.append(torch.unsqueeze(img,0))
    img=torch.cat(img_list,0)



    rgb_restored = model_restoration(img)

    # print(imgname)
    rgb_restored = torch.clamp(rgb_restored,0,1)
    rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(rgb_restored)):
        denoised_img = img_as_ubyte(rgb_restored[batch])
        retval, buffer = cv2.imencode(".png", denoised_img * 255)
        if retval:
            denoised_img = Image.open(BytesIO(buffer)).convert('RGB')
        else:
            print("保存到内存失败")
    denoised_img=processing(denoised_img,opt)
    return denoised_img



def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def processing_DIRE(img,opt,imgname):

    model=opt.diffusion_model
    diffusion=opt.diffusion
    args=opt.DIRE_args
    img = center_crop_arr(img, opt.loadSize)
    img = img.astype(np.float32) / 127.5 - 1
    img = torch.from_numpy(np.transpose(img, [2, 0, 1]))
    img_list = []
    img_list.append(torch.unsqueeze(img,0))
    img=torch.cat(img_list,0)

    
    reverse_fn = diffusion.ddim_reverse_sample_loop
    img = reshape_image(img, args.image_size)
    
    img=img.to(opt.process_device)
    model_kwargs = {}


    latent = reverse_fn(
        model,
        (1, 3, args.image_size, args.image_size),
        noise=img,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
    )
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    recons = sample_fn(
        model,
        (1, 3, args.image_size, args.image_size),
        noise=latent,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
    )

    dire = torch.abs(img - recons)
    dire = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
    dire = dire.permute(0, 2, 3, 1)
    dire = dire.contiguous()
    for i in range(len(dire)):
        retval, buffer = cv2.imencode(".png", cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        if retval:
            img_dire = Image.open(BytesIO(buffer)).convert('RGB')
        else:
            print("保存到内存失败")
    
    img_dire=processing(img_dire,opt)

   
    
    return img_dire
