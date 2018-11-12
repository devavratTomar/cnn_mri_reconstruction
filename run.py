from data_processor import DataProvider
from cnn_unet import CnnUnet, Trainer

data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=5, file_extension='.npy')

u_net = CnnUnet(x_channels=2, y_channels=2, layers=4, feature_base=32)
trainer = Trainer(u_net, batch_size=2, validation_batch_size=10)

trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output',
              5)