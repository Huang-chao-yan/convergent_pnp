**Extrapolated Plug-and-Play Three-Operator Splitting Methods for Nonconvex Optimization with Applications to Image Restoration**

This is the code of Wu Z, Huang C, Zeng T. Extrapolated Plug-and-Play Three-Operator Splitting Methods for Nonconvex Optimization with Applications to Image Restoration[J]. arXiv preprint arXiv:2403.01144, 2024. Accepted by SIAM journal on Image Science, 2024.

[paper] https://arxiv.org/pdf/2403.01144.pdf

[Prerequisites] following the training in https://github.com/samuro95/Prox-PnP

python3.7

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

torchmetrics==0.5.0


You can download our pretrained checkpoint at https://drive.google.com/file/d/1fmCBl8lV8KGezIaH4SioqwJcySDUafD1/view?usp=sharing

Please save the trained model in the ckpts directory: /GS_denoising/ckpts/Prox-DRUNet.ckpt

You can test after downloading the .ckpt file as follows. 



cd PnP_restoration

For deblurring with different noise levels (2.55, 7.65, 12.75) under blur kernel 1 in the paper

python3 deblur.py --dataset_name set3c --PnP_algo DYSdiff --noise_level_img 2.55  --noise_level_img_ini 2.55 --alpha 0.5 --alp0 0.2 

python3 deblur.py --dataset_name set3c --PnP_algo DYSdiff --noise_level_img 7.65  --noise_level_img_ini 7.65 --alpha 0.5 --alp0 0.2

python3 deblur.py --dataset_name set3c --PnP_algo DYSdiff --noise_level_img 12.75  --noise_level_img_ini 12.75 --alpha 0.5 --alp0 0.2

to see our DeTik results in Table 1 of the paper.


For super-resolution with sf 2 and 3, different noise levels (2.55, 7.65, 12.75) under blur kernel 1 in the paper

python3 SR.py --dataset_name Set5 --PnP_algo DYSdiff --noise_level_img 2.55  --noise_level_img_ini 2.55  --alp0 0.2 --sf 2

python3 SR.py --dataset_name Set5 --PnP_algo DYSdiff --noise_level_img 7.65  --noise_level_img_ini 7.65  --alp0 0.2 --sf 2

python3 SR.py --dataset_name Set5 --PnP_algo DYSdiff --noise_level_img 12.75  --noise_level_img_ini 12.75 --alp0 0.2 --sf 2

to see our DeTik results in Table 2 of the paper.




for more blur kernel results, please check lines 23 and 85 in deblur.py

dataset_name is your test data

PnP_algo is your designed algorithm

noise_level_img is the noise you add

noise_level_img_ini is for the initial noise input

alp0 is the extrapolated parameter of the algorithm



by chaoyan
20 March, 2024
