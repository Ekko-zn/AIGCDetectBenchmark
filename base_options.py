import argparse
import os
import util
import torch
#import models
#import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        
        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_sig', default='1.0')
        parser.add_argument('--jpg_method', default='pil')
        parser.add_argument('--jpg_qual', default='95')

        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--CropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--model_path',type=str,default='./weights/classifier/CNNSpot.pth',help='the path of detection model')
        # parser.add_argument('--is_single',action='store_true',help='evaluate image by image')
        parser.add_argument('--detect_method', type=str,default='CNNSpot', help='choose the detection method')
        parser.add_argument('--noise_type', type=str,default=None, help='such as jpg, blur and resize')
        
        # path of processing model
        parser.add_argument('--LNP_modelpath',type=str,default='./weights/preprocessing/sidd_rgb.pth',help='the path of LNP pre-trained model')
        parser.add_argument('--DIRE_modelpath',type=str,default='./weights/preprocessing/lsun_bedroom.pt',help='the path of DIRE pre-trained model')
        parser.add_argument('--LGrad_modelpath', type=str,default='./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth', help='the path of LGrad pre-trained model')
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        file_name = os.path.join(opt.results_dir, f'{opt.noise_type}opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test
        
        # result dir, save results and opt
        opt.results_dir=f"./results/{opt.detect_method}"
        util.mkdir(opt.results_dir)



        if print_options:
            self.print_options(opt)



        # additional

        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
