# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@163.com
Date: 2020/12/9 17:38
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.io import loadmat

from utils import (get_image_paths,
                   imread_uint,
                   augment_img,
                   uint2single,
                   single2tensor3,
                   uint2tensor3,
                   blurkernel_synthesis,
                   gen_kernel,
                   modcrop)

# from data.isp.ISP_implement import ISP


class DatasetDREDDUNDeblurring(Dataset):
    """
    Dataset of PRED_DUN for non-blind deblurring.
    """

    def __init__(self, patch_size=96, is_train=True, mode='gray'):
        super().__init__()
        self.is_train = is_train
        self.num_channels = 1 if mode == 'gray' else 3
        self.patch_size = patch_size
        self.sigma_max = 25
        self.scales = [1]  # 1 for deblurring.
        self.sf_validation = 1
        self.kernels = loadmat(os.path.join('data', 'kernels', 'kernels_12.mat'))['kernels']  # for validation
        self.count = 0

        if is_train:
            self.high_img_paths = get_image_paths(os.path.join('data', 'trainsets', mode))
        else:
            self.high_img_paths = get_image_paths(os.path.join('data', 'testsets', 'Set3'))

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:

            if self.count % 24 == 0:
                self.sf = random.choice(self.scales)
            self.count += 1

            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 12)
            if r_value >= 6:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            elif 2 < r_value < 6:
                ker_index = np.random.randint(0, 12)
                kernel = self.kernels[0, ker_index].astype(np.float64)
                kernel /= np.sum(kernel)
            else:
                sf_k = random.choice(self.scales)
                kernel = gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            # r_value = np.random.randint(0, 8)
            # if r_value > 2:
            #     ker_index = np.random.randint(8, 12)
            #     kernel = self.kernels[0, ker_index].astype(np.float64)
            #     kernel /= np.sum(kernel)
            # else:
            #     ker_index = np.random.randint(0, 8)
            #     kernel = self.kernels[0, ker_index].astype(np.float64)
            #     kernel /= np.sum(kernel)

            # ---------------------------
            # 3) noise level
            # ---------------------------
            # if np.random.randint(0, 8) == 1:
            #     noise_level = 0.0001 / 255.0
            # else:
            #     noise_level = np.random.uniform(0., self.sigma_max) / 255.0

            noise_level = np.random.uniform(0., self.sigma_max) / 255.0

            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            l_img = l_img[0::self.sf, 0::self.sf, ...]
            # add Gaussian noise
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            h_img = h_patch

        else:

            self.sf = 1  # default

            kernel = self.kernels[0, 8].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            noise_level = 2.55 / 255.0  # validation noise level
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            l_img = l_img[0::self.sf_validation, 0::self.sf_validation, ...]  # down-sampling
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)
        noise_level = torch.tensor(noise_level).view([1, 1, 1])

        return h_img, l_img, kernel, noise_level

    def __len__(self):
        return len(self.high_img_paths)


class DatasetPREDDUNSuperResolution(Dataset):
    """
    Dataset of PRED_DUN for Single Image Super Resolution.
    """

    def __init__(self, params, is_train=True):
        super().__init__()
        self.params = params
        self.is_train = is_train
        self.num_channels = params.channels_of_img
        self.mode = params.mode if self.num_channels == 3 else 'gray'
        self.patch_size = params.img_patch_size
        self.sigma_max = 25
        self.scales = [1, 2, 3, 4]  # 1 for deblurring and other's (2, 3, 4) for SISR
        self.sf_validation = 3
        self.sf = None
        self.kernels = loadmat(os.path.join('data', 'kernels', 'kernels_12.mat'))['kernels']  # for validation
        self.count = 0

        if self.is_train:
            self.high_img_paths = get_image_paths(os.path.join('data', 'trainsets', self.mode))
        else:
            self.high_img_paths = get_image_paths(os.path.join('data', 'testsets', 'Set5'))

    def __getitem__(self, index):
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:

            if self.count % self.params.batch_size == 0:
                self.sf = random.choice(self.scales)
            self.count += 1

            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))
            # mod-crop
            h_patch = modcrop(h_patch, self.sf)

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 8)
            if r_value > 3:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            else:
                kernel = gen_kernel(k_size=np.array([25, 25]))  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            # ---------------------------
            # 3) noise level
            # ---------------------------
            noise_level = np.random.uniform(0., self.sigma_max) / 255.0

            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            l_img = l_img[0::self.sf, 0::self.sf, ...]
            # add Gaussian noise
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            h_img = h_patch

        else:
            kernel = self.kernels[0, 8].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            noise_level = 2.55 / 255.0  # validation noise level
            h_img = modcrop(h_img, self.sf_validation)
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            l_img = l_img[0::self.sf_validation, 0::self.sf_validation, ...]  # down-sampling
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            self.sf = self.sf_validation

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)
        noise_level = torch.tensor(noise_level).view([1, 1, 1])
        scale_factor = self.sf

        return h_img, l_img, kernel, noise_level, scale_factor

    def __len__(self):
        return len(self.high_img_paths)


class DatasetPREDDUNRawDataDeblurring(Dataset):
    """
    Date: Nov 14, 2021.
    Dataset of PRED_DUN for non-blind deblurring of raw-data.
    """

    def __init__(self, patch_size=96, is_train=True, mode='gray'):
        super().__init__()
        self.is_train = is_train
        self.num_channels = 1 if mode == 'gray' else 3
        self.patch_size = patch_size
        self.sigma_max = 25
        self.kernels = loadmat(os.path.join('data', 'kernels', 'kernels_12.mat'))['kernels']  # for validation

        if is_train:
            self.high_img_paths = get_image_paths(os.path.join('data', 'trainsets', mode))
        else:
            self.high_img_paths = get_image_paths(os.path.join('data', 'testsets', 'Set3'))

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        # h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image
        isp = ISP()
        # To node that opencv store image in BGR,
        # When apply to color tranfer, BGR should be transfer to RGB
        h_img = cv2.imread(h_path)
        np.array(h_img, dtype='uint8')
        h_img = h_img.astype('double') / 255.0
        h_img = isp.BGR2RGB(h_img)

        if self.is_train:

            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))
            # RGB -> RAW-DATA
            h_patch, l_patch, noise_map, _ = isp.cbdnet_noise_generate_raw(h_patch)
            h_patch = np.expand_dims(h_patch, axis=2)
            l_patch = np.expand_dims(l_patch, axis=2)
            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 8)
            if r_value >= 4:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            elif 2 < r_value < 4:
                ker_index = np.random.randint(0, 12)
                kernel = self.kernels[0, ker_index].astype(np.float64)
                kernel /= np.sum(kernel)
            else:
                sf_k = 1
                kernel = gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            noise_level = np.mean(noise_map)

            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur
            l_img = ndimage.filters.convolve(l_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            # add Gaussian noise
            h_img = h_patch

        else:
            kernel = self.kernels[0, 8].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            # RGB -> RAW-DATA
            h_img, l_img, noise_map, _ = isp.cbdnet_noise_generate_raw(h_img)
            noise_level = np.mean(noise_map)  # validation noise level
            h_img = np.expand_dims(h_img, axis=2)
            l_img = np.expand_dims(l_img, axis=2)
            l_img = ndimage.filters.convolve(l_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        h_img, l_img = single2tensor3(h_img), single2tensor3(l_img)
        noise_level = torch.tensor(noise_level).view([1, 1, 1])

        return h_img, l_img, kernel, noise_level

    def __len__(self):
        return len(self.high_img_paths)
