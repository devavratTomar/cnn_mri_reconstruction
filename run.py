from data_processor import DataProvider
from cnn_deep_cascade import DeepCascade, Trainer
import create_dataset

N_EPOCHS = 20
data_provider_train = DataProvider(directory_name='./data/train', epochs=1, file_extension='.npy')
data_provider_test = DataProvider(directory_name='./data/test', epochs=0, file_extension='.npy')

opt_mask = create_dataset.get_opt_mask()

u_net = DeepCascade(x_channels=2, y_channels=2, layers=4, mask=opt_mask, feature_base=64)

trainer = Trainer(u_net, batch_size=5, validation_batch_size=50)
trainer.train(data_provider_train.get_images_iter,
              data_provider_test.get_sample_images,
              'output',
              0.8,
              N_EPOCHS,
              10)
