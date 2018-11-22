# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:47:15 2018

@author: devav
"""

import numpy as np
import matplotlib.pyplot as plt

#test = np.load('./mri1_30_2.npy')
#
#mask = test[:,:,4] + 1j*test[:,:,5]

mask2 = np.zeros((256,256))

x_ = np.random.permutation(256)[:13]
y_ = np.random.permutation(256)[:13]

mask2[x_, :] = 1.
mask2[:, y_] = 1.

img = np.fft.ifft2(mask2)
imgabs = np.abs(img)
plt.imshow(np.abs(img))