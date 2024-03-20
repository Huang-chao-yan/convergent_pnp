import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import single2uint,crop_center, imread_uint, imsave, modcrop, matlab_style_gauss2D
from natsort import os_sorted
from prox_PnP_restoration import PnP_restoration
from utils.utils_sr import numpy_degradation
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
warnings.filterwarnings("ignore")

def SR():
    parser = ArgumentParser()
    parser.add_argument('--kernel_path', type=str, default=os.path.join('kernels', 'kernels9.mat'))
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'

    if hparams.PnP_algo == 'DYSdiff':
        hparams.alpha = 0.5
        if hparams.noise_level_img == 2.55:
            hparams.lamb = 1
            hparams.gma = 0.00075
            hparams.sigma_denoiser = max(1.4 * hparams.noise_level_img, 1.9)
            
        if hparams.noise_level_img == 7.65:
            hparams.lamb = 0.9
            hparams.gma = 0.0005
            hparams.sigma_denoiser = max(0.7 * hparams.noise_level_img, 1.9)
            
        if hparams.noise_level_img == 12.75:
            hparams.lamb = 0.6
            hparams.gma = 0.0003
            hparams.sigma_denoiser = 0.6 * hparams.noise_level_img
    else:
        hparams.sigma_denoiser = max(0.5 * hparams.noise_level_img, 1.9)
        hparams.lamb = 0.99

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path, hparams.dataset_name)
    input_path = os.path.join(input_path, os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path, p) for p in os.listdir(input_path)])

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        den_out_path = 'SR'
        if not os.path.exists(den_out_path):
            os.mkdir(den_out_path)
        exp_out_path = os.path.join(den_out_path, hparams.PnP_algo)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    # Load the 8 blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']
    k_list = range(1)
    psnr_list = []
    F_list = []

    print(
        '\n Prox-PnP ' + hparams.PnP_algo + ' super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(
            hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))


    for k_index in k_list: # For each kernel

        psnr_k_list = []
        psnrY_k_list = []
        if k_index == 8: # Uniform blur
            k = (1/81)*np.ones((9,9))
        elif k_index == 9:  # Gaussian blur
            k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
        else: # Motion blur
            k = kernels[0, k_index].astype(np.float64)

        # k = kernels[0, k_index].astype(np.float64)

        if hparams.extract_images or hparams.extract_curves:
            kout_path = os.path.join(exp_out_path, 'kernel_' + str(k_index))
            if not os.path.exists(kout_path):
                os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            print('__ kernel__',k_index, '__ image__',i)

            ## load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            # Degrade image
            input_im_uint = modcrop(input_im_uint, hparams.sf)
            input_im = np.float32(input_im_uint / 255.)
            blur_im = numpy_degradation(input_im, k, hparams.sf)
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise
            init_im = blur_im

            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY,  output_ssim, output_ssimY,  x_list, z_list, y_list, Dg_list, psnr_tab, Dx_list, g_list, F_list, Psi_list = PnP_module.restore(blur_im, init_im, input_im, k, extract_results=True)
            else:
                deblur_im, output_psnr, output_psnrY = PnP_module.restore(blur_im, init_im, input_im, k)

            print('PSNR: {:.2f}dB'.format(output_psnr))

            psnr_k_list.append(output_psnr)
            psnrY_k_list.append(output_psnrY)
            psnr_list.append(output_psnr)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, psnr_tab, F_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(kout_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)
                save_im_path = os.path.join(save_im_path, hparams.PnP_algo)
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)

                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_blur.png'), single2uint(blur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(init_im))
                print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path, 'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))
        avg_k_psnrY = np.mean(np.array(psnrY_k_list))
        print('avg Y psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnrY))

    print(np.mean(np.array(psnr_list)))
    return np.mean(np.array(psnr_list))

if __name__ == '__main__':
    SR()
