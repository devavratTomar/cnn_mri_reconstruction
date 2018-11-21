"""
This module contains utilities to load data into tensorlow dataset during training and testing so that memory is not
overloaded. At a time only data required for batch processing during neural network optimization is loaded.
"""

import os
import random
import numpy as np

class DataProvider(object):
    """
    Loads the data into memory and provides an iterator for accessing the data
    
    :param directory_name: name of directory where data is present
    :param epochs: number of epochs to go over dataset
    :param file_extension: data file format
    """
    
    def __init__(self, directory_name, epochs, file_extension):
        if not os.path.isdir(directory_name):
            raise Exception("The directory name provided '{}' is incorrect.".format(directory_name))
        
        self.epochs = epochs
        self.file_names = [os.path.join(directory_name, f)\
                           for f in os.listdir(directory_name)\
                           if f.endswith(file_extension)]
        
        self.data_size = len(self.file_names)
        
    def get_sample_images(self, number_images):
        images_input = []
        images_output = []
        masks = []
        
        for file_name in self.file_names[:number_images]:
            _data = np.load(file_name)
            images_input.append(_data[:,:,[0,1]])
            images_output.append(_data[:,:,[2,3]])
            masks.append(_data[:,:,4])
        
        return np.array(images_input), np.array(images_output), np.array(masks)
    
    def get_images_iter(self, batch_size):
        """
        Returns a tensor containing batch_size images
        """
        
        n_batches = len(self.file_names)//batch_size
        for _ in range(self.epochs):
            # first shuffle the file names so that in every epoch we get different combinations.
            random.shuffle(self.file_names)
            for i in range(n_batches):
                images_input = []
                images_output = []
                batch_names = self.file_names[(batch_size*i):(batch_size*(i+1))]
                for file_name in batch_names:
                    _data = np.load(file_name)
                    images_input.append(_data[:,:,[0,1]])
                    images_output.append(_data[:,:,[2,3]])
                yield np.array(images_input), np.array(images_output)
