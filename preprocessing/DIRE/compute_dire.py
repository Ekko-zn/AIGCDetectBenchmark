"""
Modified from guided-diffusion/scripts/image_sample.py
"""

import argparse
import os
import torch

import sys
import cv2
from mpi4py import MPI

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist


sys.path.append('.')
from preprocessing_model.guided_diffusion import dist_util, logger
from preprocessing_model.guided_diffusion.image_datasets import load_data_for_reverse
from preprocessing_model.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    logger.log("have created model and diffusion")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print(args.noise_type)
    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond,
        noise_type=args.noise_type
    )
    logger.log("have created data loader")

    logger.log("computing recons & DIRE ...")
    have_finished_images = 0
    while have_finished_images < args.num_samples:
        if (have_finished_images + MPI.COMM_WORLD.size * args.batch_size) > args.num_samples and (
            args.num_samples - have_finished_images
        ) % MPI.COMM_WORLD.size == 0:
            batch_size = (args.num_samples - have_finished_images) // MPI.COMM_WORLD.size
        else:
            batch_size = args.batch_size
        all_images = []
        all_labels = []
        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]


        imgs = imgs.to(dist_util.dev())

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
        reverse_fn = diffusion.ddim_reverse_sample_loop
        imgs = reshape_image(imgs, args.image_size)

        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        recons = sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        dire = th.abs(imgs - recons)
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)
        recons = recons.contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        dire = dire.permute(0, 2, 3, 1)
        dire = dire.contiguous()

        gathered_samples = [th.zeros_like(recons) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, recons)  # gather not supported with NCCL

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        have_finished_images += len(all_images) * batch_size
        # print(th.mean(res.float()))
        recons = recons.cpu().numpy()
        for i in range(len(recons)):
            if args.has_subfolder:
                recons_save_dir = os.path.join(args.recons_dir, paths[i].split("/")[-2])
                dire_save_dir = os.path.join(args.dire_dir, paths[i].split("/")[-2])
                # dire_save_dir = os.path.join(dire_save_dir, paths[i].split("/")[-2])
                
            else:
                recons_save_dir = args.recons_dir
                dire_save_dir = args.dire_dir
            
            fn_save = os.path.basename(paths[i])
            if args.has_subclasses:
                class_name=paths[i].split("/")[-3]
                fn_save =f"{class_name}_{fn_save}"
           
            
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)
            cv2.imwrite(
                f"{dire_save_dir}/dire_{fn_save}", cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            # cv2.imwrite(f"{recons_save_dir}/{fn_save}", cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        logger.log(f"have finished {have_finished_images} samples")

    dist.barrier()
    logger.log("finish computing recons & DIRE!")


def create_argparser():
    defaults = dict(
        images_dir="/hotdata/AIGCDetect/test/progan/cow",
        recons_dir="/hotdata/AIGCDetect/test_dire/recons/stargan",
        dire_dir="/hotdata/AIGCDetect/test_dire/dire/stargan",
        noise_type="resize",
        clip_denoised=True,
        num_samples=3998,
        batch_size=16,
        use_ddim=False,
        model_path="models/lsun_bedroom.pt",
        real_step=0,  #test
        continue_reverse=False,
        has_subfolder=False,
        has_subclasses=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
