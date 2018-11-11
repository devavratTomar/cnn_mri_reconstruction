from data_processor import DataProvider
from cnn_unet import CnnUnet, Trainer

data_provider_train = DataProvider(directory_name='./data/train_sample', epochs=1, file_extension='.JPG')
data_provider_test = DataProvider(directory_name='./data/test', epochs=1, file_extension='.JPG')

u_net = CnnUnet(x_channels=3, y_channels=3, layers=3, feature_base=16)

trainer = Trainer(u_net, batch_size=1, validation_batch_size=5)

trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              './output',
              1)