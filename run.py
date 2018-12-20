from data_processor import DataProvider
from cnn_deep_cascade import DeepCascade, Trainer
from create_dataset import get_opt_mask, run_data_augmentation

import os
import sys, getopt

#TODO: remove this from final submission
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
N_EPOCHS = 25

if __name__ == "__main__":
    train_data_path = './data/train'
    test_data_path = './data/test'
    prediction_path = './prediction_deep_cascade_nc4_nlayers3'
    output_path = './output_deep_cascade_nc4_nlayers3'
    create_dataset = True
    
    try:
        options, args = getopt.getopt(sys.argv[1:], '', ['train_data_path=',
                                                          'test_data_path=',
                                                          'prediction_path=',
                                                          'model_output_path=',
                                                          'create_dataset='])
    except getopt.GetoptError:
        raise ValueError('Invaild arguments passed to run.py. Try run.py -h for help.')
        sys.exit(2)
        
    for opt, arg in options:
        if opt == 'h':
            print("Pass the arguments as stated. For default mode try command 'python run.py'")
            print("<python run.py"
                  " --train_data_path ./data/train"
                  " --test_data_path ./data/test"
                  "--prediction_path ./prediction_deep_cascade_nc4_nlayers3"
                  "--model_output_path ./output_deep_cascade_nc4_nlayers3.\n"
                  "--create_dataset 1"
                  "See ReadMe for more details.>")
            sys.exit()
        
        elif opt == '--train_data_path':
            train_data_path = arg
            
        elif opt == '--test_data_path':
            test_data_path = arg
            
        elif opt == '--prediction_path':
            prediction_path = arg
        
        elif opt == '--model_output_path':
            output_path = arg
        
        elif opt == '--model_output_path':
            output_path = arg
        
        elif opt == '--create_dataset':
            create_dataset = bool(int(arg))
        
        else:
            print('Invaild arguments passed to run.py. Try run.py -h for help.')
            sys.ext(2)
            
    if create_dataset:
        try:
            run_data_augmentation()
        except:
            raise ValueError("Please put original data at path 'data_original/'")
            sys.exit(2)
    
    data_provider_train = DataProvider(directory_name=train_data_path, epochs=1, file_extension='.npy')
    data_provider_test = DataProvider(directory_name=test_data_path, epochs=0, file_extension='.npy')
    
    opt_mask = get_opt_mask(subsample=1)
    
    deep_net = DeepCascade(x_channels=2, layers=3, ncascade=4, mask_in=opt_mask, features=64, filter_size=3)
    
    trainer = Trainer(deep_net, batch_size=15, validation_batch_size=20)
    trainer.train(data_provider_train.get_images_iter,
                  data_provider_test.get_sample_images,
                  output_path,
                  N_EPOCHS,
                  10,
                  prediction_path)
    