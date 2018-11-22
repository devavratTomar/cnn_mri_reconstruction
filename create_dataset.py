from scipy.io import loadmat
from scipy.stats import norm
import os
import shutil
import numpy as np
import logging

MAX_VALUE = 255.
TRAIN_SPLIT = 0.8
IMAGES_GEN = 10

def create_horizontal_mask(prob_params, N, keep_ratio):
    middle_index = N//2
    x_first = np.linspace(0, middle_index, middle_index+1, dtype=int)
    y_first = prob_params[0] + x_first*(prob_params[1] - prob_params[0])/(middle_index)
    
    x_second = np.linspace(middle_index+1, N-1, N-middle_index-1, dtype=int)
    y_second = prob_params[1] + (prob_params[2] - prob_params[1])*(x_second - middle_index)/(N-middle_index)
    
    prob = np.concatenate((y_first, y_second))/(np.sum(y_first) + np.sum(y_second))
    choosen_index = np.random.choice(N, int(keep_ratio*N), replace=False, p=prob)
    
    mask = np.zeros((N,N))
    mask[choosen_index, :] = 1.
    return mask

def create_horizontal_custom_mask(probs, N, keep_ratio):
    chosen_index = np.random.choice(N, int(keep_ratio*N), replace=False, p=probs/np.sum(probs))
    mask = np.zeros((N,N))
    mask[chosen_index, :] = 1
    return mask
    
def create_input_image(image, mask):
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    out = mask*image_fft    
    return np.fft.ifft2(np.fft.fftshift(out))

def create_output_image(image, mask, lambda_):
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    image_output_fft = ((lambda_)*mask + np.ones_like(mask))*image_fft
    return np.fft.ifft2(np.fft.fftshift(image_output_fft))

def load_mat_data(file_name):
    return loadmat(file_name)['data']/MAX_VALUE
    
def process_save_mat_data(images, output_folder, keep_mask_ratios, prefix_output_name):
    n_images, N, M = images.shape
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)    
    os.mkdir(output_folder)
    
    for mask_ratio in keep_mask_ratios:
        out_filename = prefix_output_name + '_Mask_' + str(mask_ratio) + '_image_'
        prob = norm.pdf(np.arange(N), N//2, mask_ratio*(N//2))
        mask = create_horizontal_custom_mask(prob, N, mask_ratio)
        for it in range(n_images):
            logging.info("Generating data for image: {nbr}, mask:{mask}, at path: {output_folder}".format(nbr=it,
                         mask=mask_ratio, output_folder=output_folder))
            for it_gen in range(IMAGES_GEN):
                output_image = np.copy(images[it, :, :])
                input_image = create_input_image(output_image, mask)
                
                input_image = input_image/np.max(np.abs(input_image))
                
                save_data_image = np.zeros((N, N, 5))
                save_data_image[:,:,0] = np.real(input_image)
                save_data_image[:,:,1] = np.imag(input_image)
                save_data_image[:,:,2] = np.real(output_image)
                save_data_image[:,:,3] = np.imag(output_image)
                save_data_image[:,:,4] = mask
                np.save(output_folder + '/' + out_filename + str(it) + '_gen_' + str(it_gen), save_data_image)
                
                
data_1 = load_mat_data('./data_original/train.mat')
data_2 = load_mat_data('./data_original/test.mat')
data_net = np.concatenate((data_1, data_2), axis=0)
n_train = int(TRAIN_SPLIT*data_net.shape[0])

process_save_mat_data(data_net[:n_train], './data/train', [0.1, 0.2], 'train')
process_save_mat_data(data_net[n_train:], './data/test', [0.1, 0.2], 'test')
