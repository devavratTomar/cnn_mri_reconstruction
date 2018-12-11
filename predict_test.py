from data_processor import DataProvider
from cnn_deep_cascade import DeepCascade
import utils
from create_dataset import get_opt_mask

data_provider_test = DataProvider(directory_name='./data/test', epochs=5, file_extension='.npy')

opt_mask = get_opt_mask(subsample=1)
deep_net = DeepCascade(x_channels=2, layers=3, ncascade=4, mask_in=opt_mask, features=64, filter_size=3)


test_x, test_y, mask = data_provider_test.get_sample_images_with_mask(10)
prediction = deep_net.predict('output_deep_cascade_nc4_nlayers3/model.ckpt', test_x)
print(prediction.shape)
utils.save_predictions_metric(test_x, test_y, prediction, mask, './prediction_test')