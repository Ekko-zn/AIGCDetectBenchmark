##  A Comprehensive Benchmark for AI-generated Image Detection [[Project Page]](https://fdmas.github.io/AIGCDetect/)


### collected methods


|method|paper|test code|train code|
|:--------:|:------:|:----:|:------:|
|CNNSpot|CNN-generated images are surprisingly easy to spot...for now|:white_check_mark:|⚙️|
|FreDect|Leveraging Frequency Analysis for Deep Fake Image Recognition|:white_check_mark:|⚙️|
|Fusing|Fusing global and local features for generalized AI-synthesized image detection|:white_check_mark:|⚙️|
|Gram-Net|Global Texture Enhancement for Fake Face Detection In the Wild|:white_check_mark:|⚙️|
|LGrad|Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection|:white_check_mark:|⚙️|
|LNP|Detecting Generated Images by Real Images|:white_check_mark:|⚙️|
|DIRE|DIRE for Diffusion-Generated Image Detection|:white_check_mark:|⚙️|
|UnivFD|Towards Universal Fake Image Detectors that Generalize Across Generative Models|:white_check_mark:|⚙️|
|RPTC|Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-generated Image Detection|⚙️|⚙️|



### Test on datasets
```
usage: eval_all.py [-h] [--rz_interp RZ_INTERP] [--blur_sig BLUR_SIG] [--jpg_method JPG_METHOD] [--jpg_qual JPG_QUAL] [--batch_size BATCH_SIZE] [--loadSize LOADSIZE] [--CropSize CROPSIZE] [--no_crop]
                   [--no_resize] [--no_flip] [--model_path MODEL_PATH] [--detect_method DETECT_METHOD] [--noise_type NOISE_TYPE] [--LNP_modelpath LNP_MODELPATH] [--DIRE_modelpath DIRE_MODELPATH]
                   [--LGrad_modelpath LGRAD_MODELPATH]

options:
  -h, --help            show this help message and exit
  --rz_interp RZ_INTERP
  --blur_sig BLUR_SIG
  --jpg_method JPG_METHOD
  --jpg_qual JPG_QUAL
  --batch_size BATCH_SIZE
                        input batch size (default: 64)
  --loadSize LOADSIZE   scale images to this size (default: 256)
  --CropSize CROPSIZE   scale images to this size (default: 224)
  --no_crop             if specified, do not crop the images for data augmentation (default: False)
  --no_resize           if specified, do not resize the images for data augmentation (default: False)
  --no_flip             if specified, do not flip the images for data augmentation (default: False)
  --model_path MODEL_PATH
                        the path of detection model (default: ./weights/CNNSpot.pth)
  --detect_method DETECT_METHOD
                        choose the detection method (default: CNNSpot)
  --noise_type NOISE_TYPE
                        such as jpg, blur and resize (default: None)
  --LNP_modelpath LNP_MODELPATH
                        the path of LNP pre-trained model (default: ./weights/sidd_rgb.pth)
  --DIRE_modelpath DIRE_MODELPATH
                        the path of DIRE pre-trained model (default: ./weights/lsun_bedroom.pt)
  --LGrad_modelpath LGRAD_MODELPATH
                        the path of LGrad pre-trained model (default: ./weights/karras2019stylegan-bedrooms-256x256_discriminator.pth)
```
:exclamation: You should set your dataroot and dataset name in `eval_config.py`


All pre-trained detection models and necessary pre-processing models are available in `./weights`

For example, if you want to evaluate the performance of CNNSpot under blurring.
```
python eval_all.py --model_path ./weights/CNNSpot.pth --detect_method CNNSpot  --noise_type blur --blur_sig 1.0 --no_resize --no_crop --batch_size 1
```

## Dataset
### Training Set
We adopt the training set in [CNNSpot](https://github.com/peterwang512/CNNDetection), you can download it from [link](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing) 

### Test Set and Checkpoints
The whole test set and checkpoints we used in our experiments can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u) or [Google Drive](https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw?usp=drive_link).


## Acknowledgments
Our code is developed based on [CNNDetection](https://github.com/peterwang512/CNNDetection), [GANDCTAnalysis](https://github.com/RUB-SysSec/GANDCTAnalysis), [Fusing](https://github.com/littlejuyan/FusingGlobalandLocal), [Gram-Net](https://github.com/liuzhengzhe/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild), [LGrad](https://github.com/chuangchuangtan/LGrad), [LNP](https://github.com/Tangsenghenshou/Detecting-Generated-Images-by-Real-Images), [DIRE](https://github.com/ZhendongWang6/DIRE), [UnivFD](https://github.com/Yuheng-Li/UniversalFakeDetect), [GenImage](https://genimage-dataset.github.io/). Thanks for their sharing codes and models.:heart:


## Citation
If you find this repository useful for your research, please consider citing this bibtex. 
```
@article{rptc,
  title={Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-generated Image Detection},
  author={Zhong, Nan and Xu, Yiran and Qian, Zhenxing and Zhang, Xinpeng},
  journal={arXiv preprint arXiv:2311.12397},
  year={2023}
}
```
