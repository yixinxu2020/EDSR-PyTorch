# NCTU_CS_T0828_HW4-Image Super Resolution
## Introduction
The proposed challenge is to train the model to reconstruct a high-resolution image from a low-resolution input, dataset with 291 training high-resolution images and 14 testing low-resolution images.
## Methodology
### Data pre-process
In the beginning, I try to write a new dataloader to load my own training data. However, I found this project has very few comments and documents about this project and it's a little cost time to understand how this project organized. Thankfully, the project is default training on DIV2K dataset, and thus, I turn to implement a simple converter to convert my dataset to DIV2K format.
```
$ python3 get_lrimg.py
```
After using **get_lrimg.py** to get myself dataloader, the structure like below:
```
datasets
├── DIV2K
│   ├── DIV2K_train_HR
│   ├── DIV2K_train_LR_bicubic
│   │   ├── X2
│   │   └── X3
│   │   └── ...
└── testing_lr_images
```
## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx 

## Training
### Clone github code
Clone this repository first.
```
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```
### Train model
Using the command below in the ./src folder to get thre first model:
```
$ python main.py --model EDSR --scale 2 --patch_size 72 --save edsr_baseline_x2 --reset
```
And than using the command below to get the final model. 
```
$ python main.py --model EDSR --scale 3 --patch_size 72 --save edsr_baseline_x3 --pre_train /home/div/cv/hw4/EDSR-PyTorch/experiment/edsr_baseline_x2/model/model_best.pt --n_GPUs 2
```
The best result in 271 epoch.
