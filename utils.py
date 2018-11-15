"""
This file contains all the utilities requried by other files.
"""

import os
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return conv_2d_b

def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def get_abs(img):
    return np.sqrt(img[:,:,0]**2 + img[:,:,1]**2)

def save_predictions(input_image, ground_truth, prediction, folder):
    for image_iter in range(input_image.shape[0]):
        img = np.concatenate((get_abs(input_image[image_iter]),
                              get_abs(ground_truth[image_iter]),
                              get_abs(prediction[image_iter])), axis=1)
        img = np.clip(img, 0, 1)
        io.imsave(os.path.join(folder, str(image_iter) + '.png'), img)
                
def save_predictions_after_transform(input_image, ground_truth, prediction, mask, lambda_, folder):
    for image_iter in range(input_image.shape[0]):
        msk = mask[image_iter]
        gt = ground_truth[image_iter]
        pd = prediction[image_iter]
        gt_cmplx = gt[:,:,0] + 1j* gt[:,:,1] # ground truth complex(image domain)
        pd_cmplx = pd[:,:,0] + 1j* pd[:,:,1] # prediction complex(image domain) 
        msk_cmplx = msk[:,:,0] + 1j* msk[:,:,1] # mask complex(image domain)
        
        bmask = np.fft.fftshift(np.fft.fft2(msk_cmplx)) # binary mask in the fourier domain
        
        # recover true ground truth image
        tr_img = (lambda_+1)*np.fft.fftshift(np.fft.fft2(gt_cmplx))/(lambda_*bmask + np.ones((gt.shape[0],gt.shape[1])))
        tr_img = np.fft.ifft2(np.fft.fftshift(tr_img))

        # recover true predicted image
        pre_img = (lambda_+1)*np.fft.fftshift(np.fft.fft2(pd_cmplx))/(lambda_*bmask + np.ones((gt.shape[0],gt.shape[1])))
        pre_img = np.fft.ifft2(np.fft.fftshift(pre_img))
        
        
        img = np.concatenate((get_abs(input_image[image_iter]),
                              np.abs(bmask),
                              np.abs(tr_img),
                              np.abs(pre_img)), axis=1)
#         plt.imshow(img)
#         plt.show()

        diff = tr_img - pre_img
        xrange = max(np.abs(pre_img).flatten())
        metrics = get_error_metrics(tr_img, pre_img)

        print('SSIM: {:.3f}, SNR: {:.1f}, PSNR: {:.2f}, l2-error: {:.3f}, l1-error: {:.3f}'.format(metrics[0], \
                                                                                              metrics[1],
                                                                                              metrics[2],
                                                                                              metrics[3],
                                                                                              metrics[4]))
        # TODO(mfsahin): save the reconstructed image to a folder and performance metrics
        # to a .csv file along with subsampling rates.
        
#         img = np.clip(img, 0, 1)
#         io.imsave(os.path.join(folder, str(image_iter) + '.png'), img)      

def get_error_metrics(f, I):
    """
    Returns the performance metrics given the true image and reconstructed image
    param(f): original image
    param(I): Reconstructed image
    """
    N = f.size
    
    # convert the images into single precision
    f = np.abs(np.float32(f)) 
    I = np.abs(np.float32(I))
    
    # TODO(mfsahin): double check this metrics
    l2_error = np.linalg.norm(f - I, 'fro')
    psnr = 20*np.log10(max(f.flatten())*np.sqrt(N)/np.linalg.norm(I - f, 'fro'))
    snr = 20*np.log10(np.linalg.norm(f.flatten())/np.linalg.norm(I - f, 'fro'))
    l1_error = np.linalg.norm(f.flatten() - I.flatten(), 1)/np.linalg.norm(f.flatten(), 1)
    ssim_ = ssim(f, I, data_range=max(I.flatten())-min(I.flatten()))

    return np.array([ssim_, snr, psnr, l2_error, l1_error])
