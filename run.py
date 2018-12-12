from data_processor import DataProvider
from cnn_deep_cascade import DeepCascade, Trainer
from create_dataset import get_opt_mask

N_EPOCHS = 100
data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=0, file_extension='.npy')

opt_mask = get_opt_mask(subsample=1)

u_net = DeepCascade(x_channels=2, layers=3, ncascade=4, mask_in=opt_mask, features=64, filter_size=3)

trainer = Trainer(u_net, batch_size=15, validation_batch_size=50)
trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output_deep_cascade_nc4_nlayers3',
              N_EPOCHS,
              10,
              prediction_path='prediction_deep_cascade_nc4_nlayers3')
