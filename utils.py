"""
This file contains all the utilities requried by other files.
"""

import os
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_mean

LAMBDA = 2

def get_variable(shape, name):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
    
def padding_circular(x, padding):
    out = tf.concat([x[:, -padding:, :, :], x, x[:, 0:padding, :, :]], axis=1)
    out = tf.concat([out[:, :, -padding:, :], out, out[:, :, 0:padding, :]], axis=2)  
    return out

def weight_variable(shape, name):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=1.))

def weight_variable_devonc(shape, name):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=1.))

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W, b, stride=1, add_custom_pad=True):
    with tf.name_scope("conv2d"):
        padding = ((tf.shape(W)[0] - 1)//2)
        if not add_custom_pad:
            conv_2d = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
        else:
            x_padded = padding_circular(x, padding)
            conv_2d = tf.nn.conv2d(x_padded, W, strides=[1, stride, stride, 1], padding='VALID')
        
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

def get_abs_complex(img):
    return np.abs(img)

def get_actual_img(ground_truth, prediction, mask):
    ac_ground_truth_fft = np.fft.fftshift(np.fft.fft2(ground_truth[:,:,0] + 1j*ground_truth[:,:,1]))
    ac_ground_truth_fft = ac_ground_truth_fft/(LAMBDA*mask + np.ones_like(mask))
    
    ac_prediction_fft = np.fft.fftshift(np.fft.fft2(prediction[:,:,0] + 1j*prediction[:,:,1]))
    ac_prediction_fft = ac_prediction_fft/(LAMBDA*mask + np.ones_like(mask))
    
    return get_abs_complex(np.fft.ifft2(np.fft.fftshift(ac_ground_truth_fft))), get_abs_complex(np.fft.ifft2(np.fft.fftshift(ac_prediction_fft)))

def save_predictions(input_image, ground_truth, prediction, masks, folder):
    for image_iter in range(input_image.shape[0]):
        img = np.concatenate((get_abs(input_image[image_iter]),
                              get_abs(ground_truth[image_iter]),
                              get_abs(prediction[image_iter])), axis=1)
        img = np.clip(img, 0, 1)
        io.imsave(os.path.join(folder, str(image_iter) + '.png'), img)
        
#        img_ac = np.concatenate(get_actual_img(ground_truth[image_iter], prediction[image_iter], masks[image_iter]), axis=1)
#        img_ac = np.clip(img_ac, 0, 1)
#        io.imsave(os.path.join(folder, str(image_iter) + '_actual_' + '.png'), img_ac)
                
def save_predictions_metric(input_image, ground_truth, prediction, mask, folder):
    metrics_avg = np.zeros(5)
    for image_iter in range(ground_truth.shape[0]):
        im = input_image[image_iter]
        gt = ground_truth[image_iter]
        pd = prediction[image_iter]
        gt_cmplx = gt[:,:,0] + 1j* gt[:,:,1] # ground truth complex(image domain)
        pd_cmplx = pd[:,:,0] + 1j* pd[:,:,1] # prediction complex(image domain) 
        im_cmplx = im[:,:,0] + 1j* im[:,:,1]
        
        img = np.concatenate((get_abs_complex(im_cmplx),
                              get_abs_complex(gt_cmplx),
                              get_abs_complex(pd_cmplx),
                              get_abs_complex(mask)), axis=1)

        metrics = get_error_metrics(gt_cmplx, pd_cmplx)
        metrics_avg = metrics_avg + metrics
        metric_str = "SSIM: {:.3f}, SNR: {:.1f}, PSNR: {:.2f}, l2-error: {:.3f}, l1-error: {:.3f}\n".format(metrics[0],\
                     metrics[1],
                     metrics[2],
                     metrics[3],
                     metrics[4])
        with open('metrics.txt','a') as f:
            f.write(str(image_iter) + ' : '+ metric_str)
        
        img = np.clip(img, 0, 1)
        io.imsave(os.path.join(folder, str(image_iter) + '.png'), img)
    
    return metrics_avg/ground_truth.shape[0]
        
def get_error_metrics(f, I):
    """
    Returns the performance metrics given the true image and reconstructed image
    param(f): original image
    param(I): Reconstructed image
    """
    # convert the images into single precision
    f = np.abs(f) 
    I = np.abs(I)
    
    # compute convex hull for better metrics
    roi = get_roi(f)
    roi_index = np.where(roi == True)
    
    f_flatten = f[roi_index]
    I_flatten = I[roi_index]
    N = f_flatten.size
    
    l2_error = np.linalg.norm(f_flatten - I_flatten)
    psnr = 20*np.log10(max(f_flatten)*np.sqrt(N)/np.linalg.norm(I_flatten - f_flatten))
    snr = 20*np.log10(np.linalg.norm(f_flatten)/np.linalg.norm(I_flatten - f_flatten))
    l1_error = np.linalg.norm(f_flatten - I_flatten, 1)/np.linalg.norm(f_flatten, 1)
    ssim_ = ssim(f_flatten, I_flatten, data_range=max(I_flatten)-min(I_flatten))

    return np.array([ssim_, snr, psnr, l2_error, l1_error])
    
def get_roi(image_):
    """
    Returns the region of interest for calculating error metrics
    param(image_): image
    param(thresh): value for binary thresholding
    """
    thresh = threshold_mean(image_)
    binary = (image_ > thresh)
    chull = convex_hull_image(binary)
    
    return chull

