from data_processor import DataProvider
from cnn_deep_cascade import DeepCascade, Trainer

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
N_EPOCHS = 25

train_data_path = './data/train'
test_data_path = './data/test'
prediction_path = './prediction_deep_cascade_nc4_nlayers3'
output_path = './output_deep_cascade_nc4_nlayers3'

data_provider_train = DataProvider(directory_name=train_data_path, epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name=test_data_path, epochs=0, file_extension='.npy')

deep_net = DeepCascade(x_channels=2, layers=2, ncascade=3, features=4, filter_size=3)
    
trainer = Trainer(deep_net, batch_size=15, validation_batch_size=20)

trainer.train(data_provider_train.get_images_iter,
                  data_provider_test.get_sample_images,
                  output_path,
                  N_EPOCHS,
                  10,
                  prediction_path=prediction_path)
    