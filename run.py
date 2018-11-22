from data_processor import DataProvider
from cnn_unet import CnnUnet, Trainer

N_EPOCHS = 2
data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=0, file_extension='.npy')

u_net = CnnUnet(x_channels=2, y_channels=2, layers=12, feature_base=16, feature_reconstruction=16)

trainer = Trainer(u_net, batch_size=5, validation_batch_size=50)
trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output',
              0.8,
              N_EPOCHS)
