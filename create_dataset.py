from scipy.io import loadmat
import scipy.ndimage as ndimage
import os
import shutil
import numpy as np

MAX_VALUE = 255.
TRAIN_SPLIT = 0.8

def get_mask(N, strip_width=8):
    n_strips = (N//2)//strip_width
    common_diff = np.arange(n_strips, 0, -1)
    sample_index = []
    for it in range(common_diff.shape[0]):
        max_index = min((it+1)*strip_width, N//2)
        sample_index.append(np.arange(it*strip_width, max_index, common_diff[it]))
    
    common_diff = np.flip(common_diff)
    for it in range(common_diff.shape[0]):
        max_index = min((it+1)*strip_width + N//2, N)
        sample_index.append(np.arange(it*strip_width + N//2, max_index, common_diff[it]))
        
    sample_index = np.delete(np.concatenate(sample_index), 0)
    
    print("Compression ratio: {}".format(float(sample_index.shape[0])/N))
    
    mask = np.zeros((N,N))
    mask[sample_index, :] = 1
    
    return mask
    
def create_input_image(image, mask):
    image_fft = np.fft.fft2(image)
    out = image_fft*np.fft.fftshift(mask)   
    return np.fft.ifft2(out)

def create_output_image(image, mask, lambda_):
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    image_output_fft = ((lambda_)*mask + np.ones_like(mask))*image_fft
    return np.fft.ifft2(np.fft.fftshift(image_output_fft))

def load_mat_data(file_name):
    return loadmat(file_name)['data']/MAX_VALUE

def augment_data(image):
    # rotate the image by 90 degress
    images = [image]
    
    for angle in range(10, 360, 10):
        images.append(ndimage.rotate(np.real(image),  angle, reshape=False) + 1j*ndimage.rotate(np.imag(image),  angle, reshape=False))
    
    # mirror flipping
    images.append(np.flip(image, axis=0))
    images.append(np.flip(image, axis=1))    
    return images

def process_save_mat_data(images, output_folder, strip_width=8):
    n_images, N, M = images.shape
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)    
    os.mkdir(output_folder)
    mask = get_mask(N, strip_width)
    
    for it in range(n_images):
        print("Generating data for image: {nbr}, at path: {output_folder}".format(nbr=it,output_folder=output_folder))        
        for aug_type , im in enumerate(augment_data(images[it, :, :])):
            output_image = np.copy(im)
            input_image = create_input_image(output_image, mask)
            
            save_data_image = np.zeros((N, N, 5))
            save_data_image[:,:,0] = np.real(input_image)
            save_data_image[:,:,1] = np.imag(input_image)
            save_data_image[:,:,2] = np.real(output_image)
            save_data_image[:,:,3] = np.imag(output_image)
            save_data_image[:,:,4] = mask
            np.save(output_folder + '/' + 'Mri_' + str(it) + '_aug_'+ str(aug_type), save_data_image)
                
                
data_1 = load_mat_data('./data_original/train.mat')
data_2 = load_mat_data('./data_original/test.mat')
data_net = np.concatenate((data_1, data_2), axis=0)

n_train = int(TRAIN_SPLIT*data_net.shape[0])

process_save_mat_data(data_net[:n_train], './data/train')
process_save_mat_data(data_net[n_train:], './data/test')
