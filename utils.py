"""
This file contains all the utilities requried by other files.
"""

import os
import numpy as np
from skimage import io
import tensorflow as tf

LAMBDA = 2

def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b, keep_prob):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob)

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
        
        img_ac = np.concatenate(get_actual_img(ground_truth[image_iter], prediction[image_iter], masks[image_iter]), axis=1)
        img_ac = np.clip(img_ac, 0, 1)
        io.imsave(os.path.join(folder, str(image_iter) + '_actual_' + '.png'), img_ac)