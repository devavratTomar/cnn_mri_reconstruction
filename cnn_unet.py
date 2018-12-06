"""
Implementation of Unet in tensorflow
"""

import tensorflow as tf
import utils
import numpy as np
from collections import OrderedDict
import logging
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#TODO: Pass image size as parameter
IMAGE_SIZE = 128

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_generator_network(x, channels_x, channels_y, layers, feature_base, keep_prob, is_train, reuse=False, create_summary=True):
    """
    :param x: input_tensor, shape should be [None, n, m, channels_x]
    :param channels_x: number of channels in the input image. For Mri, input has 4 channels.
    :param channels_y: number of channels in the output image. For Mri, output has 2 channels.
    :param layers: number of layers in u-net architecture.
    :param feature_base: Neurons in first layer of cnn. Next layers have twice the number of neurons in previous layers.
    :param keep_prob: dropout probability
    :param is_train: placeholder for telling batch normalization that we are in training phase
    :param reuse: reuse the network variables
    :create_summary: Creates Tensorboard summary if True
    """
    MINI_FILTER_SIZE = 3
    WIDE_FILTER_SIZE = 3
    
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        logging.info("Layers: {layers}, features: {features} input channels {in_channels}, output channels {out_channels}".format(
                      layers=layers,
                      features=feature_base,
                      in_channels=channels_x,
                      out_channels=channels_y))
        
        #placeholder for input image
        n = tf.shape(x)[1]
        m = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
        input_node = x_image

        dw_h_convs = OrderedDict()
        
        # down layers
        # Use filter mask of 7 for first layer of down layer and filter mask of 3 for rest of the layers.
        for layer in range(layers):
            with tf.variable_scope("down_conv_layer{}".format(str(layer))):
                features = (2**layer)*feature_base
                
                if  layer == 0:
                    std_dev = np.sqrt(2./(WIDE_FILTER_SIZE*WIDE_FILTER_SIZE*(features)))
                    w1 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, channels_x, features], std_dev, "w1")
                    w2 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE,   features, features], std_dev, "w2")
                    w3 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE,   features, features], std_dev, "w3")
                else:
                    std_dev = np.sqrt(2./(MINI_FILTER_SIZE*MINI_FILTER_SIZE*(features)))
                    w1 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, features//2, features], std_dev, "w1")
                    w2 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE,    features, features], std_dev, "w2")
                    w3 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE,    features, features], std_dev, "w3")
                
                b1 = utils.bias_variable([features], "b1")
                b2 = utils.bias_variable([features], "b2")
                b3 = utils.bias_variable([features], "b3")
                
                if layer == 0:
                    conv_1 = utils.conv2d(input_node, w1, b1, keep_prob, stride=1)
                else:
                    conv_1 = utils.conv2d(input_node, w1, b1, keep_prob, stride=2)
                
                conv_1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_1, training=is_train))
                
                conv_2 = utils.conv2d(conv_1, w2, b2, keep_prob, stride=1)
                conv_2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_2, training=is_train))
                
                conv_3 = utils.conv2d(conv_2, w3, b3, keep_prob, stride=1)
                conv_3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_3, training=is_train))
                
                dw_h_convs[layer] = conv_3
                input_node = dw_h_convs[layer]
    
        #up layers
        for layer in range(layers - 2, -1, -1):
            with tf.variable_scope("up_conv_layer{}".format(str(layer))):
                features = (2**(layer + 1))*feature_base
                std_dev = np.sqrt(2./(MINI_FILTER_SIZE*MINI_FILTER_SIZE*features))
                
                w1 = utils.weight_variable_devonc([MINI_FILTER_SIZE, MINI_FILTER_SIZE, features//2, features], std_dev, "w1")
                b1 = utils.bias_variable([features//2], "b1")
                
                h_deconv = utils.deconv2d(input_node, w1, 2) + b1
                h_deconv = tf.nn.leaky_relu(tf.layers.batch_normalization(h_deconv, training=is_train))
                
               # h_deconv_sum = 0.5*tf.add(dw_h_convs[layer], h_deconv)
                h_deconv_concat = tf.concat([dw_h_convs[layer], h_deconv], axis=3)
                w2 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, features, features//2], std_dev, "w2")
                w3 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, features//2, features//2], std_dev, "w3")
                b2 = utils.bias_variable([features//2], "b2")
                b3 = utils.bias_variable([features//2], "b3")
                
                conv_2 = utils.conv2d(h_deconv_concat, w2, b2, keep_prob, stride=1)
                conv_2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_2, training=is_train))
                
                conv_3 = utils.conv2d(conv_2, w3, b3, keep_prob, stride=1)
                conv_3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_3, training=is_train))
                
                input_node = conv_3
            
        weight = utils.weight_variable([1, 1, feature_base, channels_y], std_dev, "out_weight")
        bias = utils.bias_variable([channels_y], "out_bias")
        
        output_image = utils.conv2d(input_node, weight, bias, tf.constant(1.0), stride=1, add_custom_pad=False)
        
        #TODO: Should we add input to the final reconstruction?
        output_image = tf.nn.tanh(output_image)
        
        output_image_complex = tf.complex(output_image[:, :, :, 0], output_image[:, :, :, 1])
        output_image_complex_fft = tf.spectral.fft2d(output_image_complex)
        output_image_complex_fft = tf.reshape(output_image_complex_fft, tf.stack([-1, n, m, 1]))
        
        output_image_corrected = tf.concat([tf.real(output_image_complex_fft), tf.imag(output_image_complex_fft)], axis=3)
        
        return output_image, output_image_corrected


def create_discriminator_network(x, channels_x, feature_base, keep_prob, is_train, reuse=False, create_summary=True):
    MINI_FILTER_SIZE = 3
    WIDE_FILTER_SIZE = 7
    
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        n = tf.shape(x)[1]
        m = tf.shape(x)[2]
        
        x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
        input_node = x_image
        
        ## First Layer has 3 sub layer, feature_base neurons each layer, stride 2 
        ## Second Layer has 2 sub layer, 2*feature_base neurons each layer
        ## Third Layer has 1 sub layer, 4* features_base neurons each layer
        ## Final conv layer, 4*features_base neurons each layer
        
        ## 2 FC layers
        
        with tf.variable_scope("conv_layer"):
            std_dev = np.sqrt(2./(WIDE_FILTER_SIZE*WIDE_FILTER_SIZE*feature_base))
            
            # Layer 0
            w1_0 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, channels_x, feature_base], std_dev, "w1_layer0")
            b1_0 = utils.bias_variable([feature_base], "b1_layer0")
            
            w2_0 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, feature_base, feature_base], std_dev, "w2_layer0")
            b2_0 = utils.bias_variable([feature_base], "b2_layer0")
            
            w3_0 = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, feature_base, feature_base], std_dev, "w3_layer0")
            b3_0 = utils.bias_variable([feature_base], "b3_layer0")
            
            conv1_0 = utils.conv2d(input_node, w1_0, b1_0, keep_prob, stride=1)
            conv1_0 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1_0, training=is_train))
            
            conv2_0 = utils.conv2d(conv1_0, w2_0, b2_0, keep_prob, stride=1)
            conv2_0 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2_0, training=is_train))
            
            conv3_0 = utils.conv2d(conv2_0, w3_0, b3_0, keep_prob, stride=1)
            conv3_0 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3_0, training=is_train))
            
            # max pool and reduce dimension by 2
            pool0 = utils.max_pool(conv3_0, 2)
            
            # Layer 1
            std_dev = np.sqrt(2./(MINI_FILTER_SIZE*MINI_FILTER_SIZE*2*feature_base))
            
            w1_1 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, feature_base, 2*feature_base], std_dev, "w1_layer1")
            b1_1 = utils.bias_variable([2*feature_base], "b1_layer1")            

            w2_1 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 2*feature_base, 2*feature_base], std_dev, "w2_layer1")
            b2_1 = utils.bias_variable([2*feature_base], "b2_layer1")
            
            conv1_1 = utils.conv2d(pool0, w1_1, b1_1, keep_prob, stride=1)
            conv1_1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1_1, training=is_train))
            
            conv2_1 = utils.conv2d(conv1_1, w2_1, b2_1, keep_prob, stride=1)
            conv2_1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2_1, training=is_train))
            
            # max pool and reduce dimension by 2
            pool1 = utils.max_pool(conv2_1, 2)
            
            # Layer 2
            std_dev = np.sqrt(2./(MINI_FILTER_SIZE*MINI_FILTER_SIZE*4*feature_base))
            
            w1_2 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 2*feature_base, 4*feature_base], std_dev, "w1_layer2")
            b1_2 = utils.bias_variable([4*feature_base], "b1_layer2")
            
            conv1_2 = utils.conv2d(pool1, w1_2, b1_2, keep_prob, stride=1)
            conv1_2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1_2, training=is_train))
            
            #max pool and reduce dimension by 2
            pool2 = utils.max_pool(conv1_2, 2)
            
            # Layer 3
            std_dev = np.sqrt(2./(MINI_FILTER_SIZE*MINI_FILTER_SIZE*feature_base))
            w1_3 = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 4*feature_base, feature_base], std_dev, "w1_layer3")
            b1_3 = utils.bias_variable([feature_base], "b1_layer3")
            
            conv1_3 = utils.conv2d(pool2, w1_3, b1_3, keep_prob, stride=1)
            conv1_3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1_3, training=is_train))
            
            # max pool and reduce dimension by 2
            pool3 = utils.max_pool(conv1_3, 2)
            
            num_features = int((IMAGE_SIZE*IMAGE_SIZE)*feature_base/(2**8))
            
            # FC 1
            flatten_layer = tf.reshape(pool3, [-1, num_features])
            
            w_fc_1 = utils.get_variable([num_features, num_features//4], "w_fc_1")
            b_fc_1 = utils.bias_variable([num_features//4], "b_fc_1")
            
            out_fc1 = tf.nn.leaky_relu(tf.matmul(flatten_layer, w_fc_1) + b_fc_1)
            
            # FC 2
            w_fc_2 = utils.get_variable([num_features//4, 1], "w_fc_2")
            b_fc_2 = utils.bias_variable([1], "b_fc_2")
            
            out_fc2 = tf.nn.leaky_relu(tf.matmul(out_fc1, w_fc_2) + b_fc_2)
            
            return out_fc2
            
class CnnUnet_GAN(object):
    """
    Implementation of Unet.
    
    :param x_channels: number of channels in input image
    :param y_channels: number of channels in output image
    """
    
    def __init__(self, x_channels, y_channels, layers_gen=3, feature_base_gen=64, feature_base_disc=64, create_summary=True):
        tf.reset_default_graph()
        
        self.x =    tf.placeholder("float", shape=[None, None, None, x_channels], name="x")
        self.y =    tf.placeholder("float", shape=[None, None, None, y_channels], name="y")
        self.mask = tf.placeholder("float", shape=[None, None, None, 1], name="mask")
        
        self.keep_prob = tf.placeholder("float", name="dropout_probability")
        self.is_train = tf.placeholder("bool", name="batch_norm_is_train")
        
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.create_summary = create_summary
        
        fake_output, fake_output_fft = create_generator_network(x= self.x,
                                                                channels_x= x_channels,
                                                                channels_y= y_channels,
                                                                layers= layers_gen,
                                                                feature_base= feature_base_gen,
                                                                keep_prob=self.keep_prob,
                                                                is_train=self.is_train,
                                                                create_summary= create_summary)
        
        
        real_logit = create_discriminator_network(self.y,
                                                  y_channels,
                                                  feature_base=feature_base_disc,
                                                  keep_prob=self.keep_prob,
                                                  is_train=self.is_train)
        
        fake_logit = create_discriminator_network(fake_output,
                                                  y_channels,
                                                  feature_base=feature_base_disc,
                                                  keep_prob=self.keep_prob,
                                                  is_train=self.is_train,
                                                  reuse=True)
        
        self.cost_discriminator = self.__get_cost_discriminator(real_logit, fake_logit)
        self.cost_generator = self.__get_cost_generator(fake_logit, fake_output, fake_output_fft)
        
        self.generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
        self.discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
        
        self.discriminator_acc_fake, self.discriminator_acc_real = self.__get_discriminator_accuracy(fake_logit, real_logit)
        
        with tf.name_scope("resuts"):
            self.predictor = fake_output
    
    def __get_discriminator_accuracy(self, fake_logit, real_logit):
        prob_fake = tf.sigmoid(fake_logit)
        prob_real = tf.sigmoid(real_logit)
        
#        acc_fake = tf.reduce_mean(tf.cast(tf.greater(0.5, prob_fake), float))
#        acc_real = tf.reduce_mean(tf.cast(tf.greater(prob_real, 0.5), float))
        
        acc_fake = 1.0 - tf.reduce_mean(prob_fake)
        acc_real = tf.reduce_mean(prob_real)
        return acc_fake, acc_real
    
    def __get_cost_discriminator(self, real_logit, fake_logit):
        with tf.name_scope("cost_discriminator"):
            loss = tf.reduce_mean(\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,labels=tf.random_uniform(tf.shape(real_logit), minval=0.7, maxval=1.2)) +\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,labels=tf.random_uniform(tf.shape(fake_logit), minval=0.0, maxval=0.3)))
            
            return loss
        
    def __get_cost_generator(self, fake_logit, fake_output, fake_output_fft):
        with tf.name_scope("cost_generator"):
            loss_mse_image = tf.losses.mean_squared_error(self.y, fake_output)
            
            input_image_cmplx = tf.complex(self.x[:, :, :, 0], self.x[:, :, :, 1])
            input_image_fft = tf.reshape(tf.spectral.fft2d(input_image_cmplx), tf.stack([-1, IMAGE_SIZE, IMAGE_SIZE, 1]))
            
            loss_mse_fft = tf.losses.mean_squared_error(tf.concat([tf.real(input_image_fft), tf.imag(input_image_fft)], axis=3),\
                                                        fake_output_fft*self.mask)
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.random_uniform(tf.shape(fake_logit), minval=0.7, maxval=1.2)))
        
        return loss_gen + 10*loss_mse_image + loss_mse_fft
    
    
    def predict(self, model_path, test_image, test_mask):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((test_image.shape[0], test_image.shape[1], test_image.shape[2], self.y_channels))
            prediction = sess.run(self.predictor,
                                  feed_dict={self.x: test_image,
                                             self.y: y_dummy,
                                             self.mask: test_mask,
                                             self.keep_prob: 1.0,
                                             self.is_train: False})
            
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: {}".format(model_path))
        
        
class Trainer(object):
    """
    This will train a u-net instance.
    
    :param net: unet to train
    :param batch_size: size of training batch
    :param validation_batch_size: size of validation batch
    :param create_train_summary: add training summaries if True (e.g. gradients)
    """
    
    def __init__(self, net, batch_size=1, validation_batch_size=1, create_train_summary=True):
        self.net = net
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.create_train_summary = create_train_summary
        
        # we choose adam optimezer for this problem.
        learning_rate_dis = 0.001
        learning_rate_gen = 0.0001
        
        self.learning_rate_node_gen = tf.Variable(learning_rate_gen, name="learning_rate_generator")
        self.learning_rate_node_dis = tf.Variable(learning_rate_dis, name="learning_rate_discriminator")
    
    def __get_optimizers(self, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node_gen).minimize(self.net.cost_generator, var_list= self.net.generator_vars, global_step=global_step)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node_dis).minimize(self.net.cost_discriminator, var_list= self.net.discriminator_vars, global_step=global_step)
        return optimizer_gen, optimizer_dis
    
    def __initialize(self, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step_generator")
        
        tf.summary.scalar("loss_generator", self.net.cost_generator)
        tf.summary.scalar("loss_discriminator", self.net.cost_discriminator)
        tf.summary.scalar("discriminator_accuracy_fake", self.net.discriminator_acc_fake)
        tf.summary.scalar("discriminator_accuracy_real", self.net.discriminator_acc_real)
        
        self.optimizer_generator, self.optimizer_discriminator = self.__get_optimizers(global_step)
        
        tf.summary.scalar("learning_rate_generator", self.learning_rate_node_gen)
        tf.summary.scalar("learning_rate_discriminator", self.learning_rate_node_dis)
        
        
        self.summary_all = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        
        prediction_path_abs = os.path.abspath(prediction_path)
        output_path_abs = os.path.abspath(output_path)
        if not restore:
            if os.path.exists(prediction_path_abs):
                logging.info("Removing '{:}'".format(prediction_path_abs))
                shutil.rmtree(prediction_path_abs, ignore_errors=True)
            if os.path.exists(output_path_abs):
                logging.info("Removing '{:}'".format(output_path_abs))
                shutil.rmtree(output_path_abs, ignore_errors=True)
            
        if not os.path.exists(prediction_path_abs):
            logging.info("Creating '{:}'".format(prediction_path_abs))
            os.mkdir(prediction_path_abs)
            
        if not os.path.exists(output_path_abs):
            logging.info("Creating '{:}'".format(output_path_abs))
            os.mkdir(output_path_abs)
            
        return init
    
    def store_prediction(self, sess, batch_x, batch_y, masks, name):
        prediction = sess.run(self.net.predictor,
                              feed_dict= {self.net.x: batch_x,
                                          self.net.y: batch_y,
                                          self.net.mask: masks,
                                          self.net.keep_prob: 1.,
                                          self.net.is_train: False})
        
        loss_gen, loss_disc = sess.run((self.net.cost_generator, self.net.cost_discriminator),
                                       feed_dict= {self.net.x: batch_x,
                                                   self.net.y: batch_y,
                                                   self.net.mask: masks,
                                                   self.net.keep_prob: 1.,
                                                   self.net.is_train: False})
        
        logging.info("Generator loss = {}, Discriminator loss = {}".format(loss_gen, loss_disc))
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
        
        os.mkdir(prediction_folder)
        utils.save_predictions_metric(batch_x, batch_y, prediction, masks, prediction_folder)
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_mask):
        # Calculate batch loss and accuracy
        summary_str, loss_gen, loss_disc, disc_acc_fake, disc_acc_real = \
        sess.run((self.summary_all, self.net.cost_generator, self.net.cost_discriminator, self.net.discriminator_acc_fake, self.net.discriminator_acc_real),
                 feed_dict={self.net.x: batch_x,
                            self.net.y: batch_y,
                            self.net.mask: batch_mask,
                            self.net.keep_prob: 1.,
                            self.net.is_train: False})
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss: Generator = {:.16f}, Discriminator = {:.16f}, Accuracy fake = {}, Accuracy real = {}"\
                     .format(step, loss_gen, loss_disc, disc_acc_fake, disc_acc_real))
        
    def output_epoch_stats(self, epoch, loss_gen, loss_disc, lr_gen, lr_disc):
        logging.info(
            "Epoch {:}, Loss Generator = {:.16f}, Loss Discriminator = {:.16f}, learning rate generator: {:.8f}"
            ", learning rate discriminator: {:.8f}".format(epoch, loss_gen, loss_disc, lr_gen, lr_disc))
    
    def train(self, data_provider_train, data_provider_validation,
              output_path,
              keep_prob,
              epochs=10,
              display_step=1,
              lr_update=20,
              restore=False,
              write_graph=True,
              prediction_path='prediction'):
        """
        Start training the network
        
        :param data_provider_train: callable returning training data
        :param data_provider_validation: callable returning validation data
        :param output_path: path where to store checkpoints
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        self.prediction_path = os.path.abspath(os.path.join('.', prediction_path))
        
        save_path = os.path.join(output_path, "model.ckpt")
        init = self.__initialize(output_path, restore, prediction_path)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb")
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            test_x, test_y, masks = data_provider_validation(self.validation_batch_size)
            self.store_prediction(sess, test_x, test_y, masks, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Training Started")
            
            loss_disc = 1.9
            loss_gen =  3.0
            step_counter = 0
            for epoch in range(epochs):
                print(epoch)
                for step, (batch_x, batch_y, batch_mask) in enumerate(data_provider_train(self.batch_size)):
                    _, loss_gen, lr_gen = sess.run((self.optimizer_generator, self.net.cost_generator, self.learning_rate_node_gen),
                                                   feed_dict= {self.net.x: batch_x,
                                                               self.net.y: batch_y,
                                                               self.net.mask: batch_mask,
                                                               self.net.keep_prob: keep_prob,
                                                               self.net.is_train: True})
                    
                    _, loss_disc, lr_disc, dic_fake_acc, dic_real_acc = sess.run((self.optimizer_discriminator, self.net.cost_generator, self.learning_rate_node_dis, self.net.discriminator_acc_fake, self.net.discriminator_acc_real),
                                                                                feed_dict= {self.net.x: batch_x,
                                                                                        self.net.y: batch_y,
                                                                                        self.net.mask: batch_mask,
                                                                                        self.net.keep_prob: keep_prob,
                                                                                        self.net.is_train: True})
    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, batch_y, batch_mask)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss_gen, loss_disc, lr_gen, lr_disc)
                self.store_prediction(sess, test_x, test_y, masks, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
                
                if epoch % lr_update == 0 and epoch != 0:
                    sess.run(self.learning_rate_node.assign(self.learning_rate_node.eval()/2))
            
            summary_writer.close()
        logging.info("Training Finished")
        return save_path
