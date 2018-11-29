from data_processor import DataProvider
from cnn_unet import CnnUnet
import utils

data_provider_test = DataProvider(directory_name='./data/test', epochs=5, file_extension='.npy')

u_net = CnnUnet(x_channels=2, y_channels=2, layers=4, feature_base=64)


# data_provider_test.get_sample_images
test_x, test_y, mask = data_provider_test.get_sample_images(100)
prediction = u_net.predict('output_unet_version_2/model.ckpt', test_x)
print(prediction.shape)
utils.save_predictions_after_transform(test_x, test_y, prediction, './prediction_test')