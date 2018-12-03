from data_processor import DataProvider
from cnn_unet import CnnUnet_GAN
import utils

data_provider_test = DataProvider(directory_name='./data/train', epochs=5, file_extension='.npy')

u_net = CnnUnet_GAN(x_channels=2, y_channels=2)


# data_provider_test.get_sample_images
test_x, test_y, mask = data_provider_test.get_sample_images(10)
prediction = u_net.predict('output_gan_without_bn/model.ckpt', test_x, mask)
print(prediction.shape)
utils.save_predictions_metric(test_x, test_y, prediction, mask, './prediction_train')