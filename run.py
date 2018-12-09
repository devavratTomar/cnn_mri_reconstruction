from data_processor import DataProvider
from cnn_unet import CnnResnet, Trainer

N_EPOCHS = 50
data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=0, file_extension='.npy')

u_net = CnnResnet(x_channels=2, y_channels=2, layers=5)

trainer = Trainer(u_net, batch_size=15, validation_batch_size=50)
trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output_resnet_batch_norm_diff_mask',
              N_EPOCHS,
              10,
              lr_update=25)
