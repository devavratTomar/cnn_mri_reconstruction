from data_processor import DataProvider
from cnn_unet import CnnUnet_GAN, Trainer

N_EPOCHS = 100
data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=0, file_extension='.npy')

u_net = CnnUnet_GAN(x_channels=2, y_channels=2, layers_gen=4, layers_disc=6, feature_base_gen=64, feature_base_disc=64)

trainer = Trainer(u_net, batch_size=5, validation_batch_size=50)
trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output_gan_version_1',
              0.8,
              N_EPOCHS,
              10,
              lr_update=30)
