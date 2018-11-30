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

#TODO: Pass image size as parameter
IMAGE_SIZE = 256

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_generator_network(x, channels_x, channels_y, layers=4, feature_base=64, keep_prob=0.8, reuse=False, create_summary=True):
    """
    :param x: input_tensor, shape should be [None, n, m, channels_x]
    :param channels_x: number of channels in the input image. For Mri, input has 4 channels.
    :param channels_y: number of channels in the output image. For Mri, output has 2 channels.
    :param layers: number of layers in u-net architecture.
    :param feature_base: Neurons in first layer of cnn. Next layers have twice the number of neurons in previous layers.
    :param filter_size: size of convolution filter
    :param pool_size: size of pooling layer
    :create_summary: Creates Tensorboard summary if True
    """
    FILTER_SIZE_DEF = 3
    FILTER_SIZE_DIL = 5
    
    with tf.device('/device:GPU:1'):
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
            # w1 and w2 operates on input layer. w1 is size 3 and w2 is size 5 with dilation 2. w3 operates on concatenation of w1 and w2.
            for layer in range(layers):
                with tf.variable_scope("down_conv_layer{}".format(str(layer))):
                    features = (2**layer)*feature_base
                    std_dev_1 = np.sqrt(2./(FILTER_SIZE_DEF*FILTER_SIZE_DEF*(features//2)))
                    std_dev_2 = np.sqrt(2./(FILTER_SIZE_DIL*FILTER_SIZE_DIL*(features//2)))
                    std_dev_3 = np.sqrt(2./(FILTER_SIZE_DEF*FILTER_SIZE_DEF*features))
                    
                    if  layer == 0:
                        w1 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, channels_x,  features//2], std_dev_1, "w1")
                        w2 = utils.weight_variable([FILTER_SIZE_DIL, FILTER_SIZE_DIL, features//2, features], std_dev_2, "w2")
                    else:
                        w1 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features//2, features//2], std_dev_1, "w1")
                        w2 = utils.weight_variable([FILTER_SIZE_DIL, FILTER_SIZE_DIL, features//2, features], std_dev_2, "w2")
                        
                    w3 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features, features], std_dev_3, "w3")
                    
                    b1 = utils.bias_variable([features//2], "b1")
                    b2 = utils.bias_variable([features], "b2")
                    b3 = utils.bias_variable([features], "b3")
                    
                    if layer == 0:
                        conv_1 = tf.nn.leaky_relu(utils.conv2d(input_node, w1, b1, keep_prob, stride=1))
                    else:
                        conv_1 = tf.nn.leaky_relu(utils.conv2d(input_node, w1, b1, keep_prob, stride=2))
                    
    #                conv_2 = tf.nn.leaky_relu(utils.conv2d_dilated(conv_1, w2, b2, keep_prob, dilation=2))
                    conv_2 = tf.nn.leaky_relu(utils.conv2d(conv_1, w2, b2, keep_prob, stride=1))
                    conv_3 = tf.nn.leaky_relu(utils.conv2d(conv_2, w3, b3, keep_prob, stride=1))
                    
                    dw_h_convs[layer] = conv_3
                    input_node = dw_h_convs[layer]
        
            #up layers
            for layer in range(layers - 2, -1, -1):
                with tf.variable_scope("up_conv_layer{}".format(str(layer))):
                    features = (2**(layer + 1))*feature_base
                    std_dev = np.sqrt(2./(FILTER_SIZE_DEF*FILTER_SIZE_DEF*features))
                    
                    w1 = utils.weight_variable_devonc([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features//2, features], std_dev, "w1")
                    b1 = utils.bias_variable([features//2], "b1")
                    
                    h_deconv = tf.nn.leaky_relu(utils.deconv2d(input_node, w1, 2) + b1)
                    h_deconv_sum = tf.add(dw_h_convs[layer], h_deconv)
                    
                    w2 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features//2, features//2], std_dev, "w2")
                    w3 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features//2, features//2], std_dev, "w3")
                    b2 = utils.bias_variable([features//2], "b2")
                    b3 = utils.bias_variable([features//2], "b3")
                    
                    conv_2 = utils.conv2d(h_deconv_sum, w2, b2, keep_prob, stride=1)
                    conv_3 = utils.conv2d(tf.nn.leaky_relu(conv_2), w3, b3, keep_prob, stride=1)
                    
                    input_node = tf.nn.leaky_relu(conv_3)
                
            weight = utils.weight_variable([1, 1, feature_base, channels_y], std_dev, "out_weight")
            bias = utils.bias_variable([channels_y], "out_bias")
            
            output_image = tf.add(utils.conv2d(input_node, weight, bias, tf.constant(1.0), stride=1, add_custom_pad=False), x_image)
            return output_image


def create_discriminator_network(x, channels_x, layers=6, feature_base=16, keep_prob=0.8, reuse=False, create_summary=True):
    FILTER_SIZE_DEF = 3
    FILTER_SIZE_DIL = 5
    
    with tf.device('/device:GPU:0'):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            n = tf.shape(x)[1]
            m = tf.shape(x)[2]
            
            x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
            input_node = x_image
            
            for layer in range(layers):
                with tf.variable_scope("conv_layer{}".format(str(layer))):
                    features = (2**layer)*feature_base
                    std_dev_1 = np.sqrt(2./(FILTER_SIZE_DEF*FILTER_SIZE_DEF*(features//2)))
                    std_dev_2 = np.sqrt(2./(FILTER_SIZE_DIL*FILTER_SIZE_DIL*(features//2)))
                    std_dev_3 = np.sqrt(2./(FILTER_SIZE_DEF*FILTER_SIZE_DEF*features))
                    
                    if  layer == 0:
                        w1 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, channels_x,  features//2], std_dev_1, "w1")
                        w2 = utils.weight_variable([FILTER_SIZE_DIL, FILTER_SIZE_DIL, features//2, features], std_dev_2, "w2")
                    else:
                        w1 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features, features//2], std_dev_1, "w1")
                        w2 = utils.weight_variable([FILTER_SIZE_DIL, FILTER_SIZE_DIL, features//2, features], std_dev_2, "w2")
                        
                    w3 = utils.weight_variable([FILTER_SIZE_DEF, FILTER_SIZE_DEF, features, features], std_dev_3, "w3")
                    
                    b1 = utils.bias_variable([features//2], "b1")
                    b2 = utils.bias_variable([features], "b2")
                    b3 = utils.bias_variable([features], "b3")
                    
                    if layer == 0:
                        conv_1 = tf.nn.leaky_relu(utils.conv2d(input_node, w1, b1, keep_prob, stride=1))
                    else:
                        conv_1 = tf.nn.leaky_relu(utils.conv2d(input_node, w1, b1, keep_prob, stride=2))
                    
                    conv_2 = tf.nn.leaky_relu(utils.conv2d(conv_1, w2, b2, keep_prob, stride=1))
                    conv_3 = tf.nn.leaky_relu(utils.conv2d(conv_2, w3, b3, keep_prob, stride=1))
                    input_node = conv_3
            
            num_features = (IMAGE_SIZE//(2**layer))*(IMAGE_SIZE//(2**layer))*features
            # flatten final layer and add FC layer
            with tf.variable_scope("FC_layer"):
                final_flatten_input = tf.reshape(input_node, [-1, num_features])
    
                w_fc_1 = utils.get_variable([num_features, num_features//2], "w_fc_1")
                b_fc_1 = utils.bias_variable([num_features//2], "b_fc_1")
                logit_fc_1 = tf.nn.leaky_relu(tf.matmul(final_flatten_input, w_fc_1) + b_fc_1)
                
                w_fc_2 = utils.get_variable([num_features//2, 1], "w_fc_2")
                b_fc_2 = utils.bias_variable([1], "b_fc_2")
                logit_fc_2 = tf.matmul(logit_fc_1, w_fc_2) + b_fc_2
            
            return logit_fc_2

            
class CnnUnet_GAN(object):
    """
    Implementation of Unet.
    
    :param x_channels: number of channels in input image
    :param y_channels: number of channels in output image
    """
    
    def __init__(self, x_channels, y_channels, layers_gen, layers_disc, feature_base_gen, feature_base_disc, create_summary=True):
        tf.reset_default_graph()
        
        self.x =    tf.placeholder("float", shape=[None, None, None, x_channels], name="x")
        self.y =    tf.placeholder("float", shape=[None, None, None, y_channels], name="y")
        self.mask = tf.placeholder("float", shape=[None, None, None], name="mask")
        
        self.keep_prob = tf.placeholder("float", name="dropout_probability")
        
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.create_summary = create_summary
        
        fake_output= create_generator_network(x= self.x,
                                              channels_x= x_channels,
                                              channels_y= y_channels,
                                              layers= layers_gen,
                                              feature_base= feature_base_gen,
                                              keep_prob=self.keep_prob,
                                              create_summary= create_summary)
        
        
        real_logit = create_discriminator_network(self.y, y_channels, layers=layers_disc, feature_base=feature_base_disc, keep_prob=self.keep_prob)
        fake_logit = create_discriminator_network(fake_output, y_channels, layers=layers_disc, feature_base=feature_base_disc, keep_prob=self.keep_prob, reuse=True)
        
        self.cost_discriminator = self.__get_cost_discriminator(real_logit, fake_logit)
        self.cost_generator = self.__get_cost_generator(fake_logit, fake_output)
        
        self.generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
        self.discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

        self.gradients_node_generator = tf.gradients(self.cost_generator, self.generator_vars)
        self.gradients_node_discriminator = tf.gradients(self.cost_discriminator, self.discriminator_vars)
        
        with tf.name_scope("resuts"):
            self.predictor = fake_output
    
    def __get_cost_discriminator(self, real_logit, fake_logit):
        with tf.name_scope("cost_discriminator"):
            loss = tf.reduce_mean(\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,labels=tf.ones_like(real_logit)) +\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,labels=tf.zeros_like(fake_logit)))
            
            return loss
        
    def __get_cost_generator(self, fake_logit, fake_output):
        with tf.name_scope("cost_generator"):
            loss_mse = tf.losses.mean_squared_error(self.y, fake_output, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)))
        return loss_gen + loss_mse
    
    
    def predict(self, model_path, test_image, test_mask):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((test_image.shape[0], test_image.shape[1], test_image.shape[2], self.y_channels))
            prediction = sess.run(self.predictor, feed_dict={self.x: test_image, self.y: y_dummy, self.mask: test_mask, self.keep_prob: 1.0})
            
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
        learning_rate = 0.0001
        self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
    
    def __get_optimizer_generator(self, global_step):        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                            .minimize(self.net.cost_generator, var_list= self.net.generator_vars ,global_step=global_step)

        return optimizer
    
    def __get_optimizer_discriminator(self, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                            .minimize(self.net.cost_discriminator, var_list= self.net.discriminator_vars, global_step=global_step)
        return optimizer
    
    def __initialize(self, output_path, restore, prediction_path):
        global_step_gen = tf.Variable(0, name="global_step_generator")
        global_step_disc= tf.Variable(0, name="globa_step_discriminator")
        
        self.norm_gradients_node_gen = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node_generator)]), name="norm_gradient_generator")
        self.norm_gradients_node_disc= tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node_discriminator)]), name="norm_gradient_discriminator")
        
        if self.net.create_summary and self.create_train_summary:
            tf.summary.histogram("norm_gradient_generator", self.norm_gradients_node_gen)
            tf.summary.histogram("norm_gradient_discriminator", self.norm_gradients_node_disc)
            
        tf.summary.scalar("loss_generator", self.net.cost_generator)
        tf.summary.scalar("loss_discriminator", self.net.cost_discriminator)
        
        self.optimizer_generator = self.__get_optimizer_generator(global_step_gen)
        self.optimizer_discriminator = self.__get_optimizer_discriminator(global_step_disc)
        
        tf.summary.scalar("learning_rate", self.learning_rate_node)
        
        
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
        prediction = sess.run(self.net.predictor, feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.mask: masks, self.net.keep_prob: 1.})
        loss_gen, loss_disc = sess.run((self.net.cost_generator, self.net.cost_discriminator), feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.mask: masks, self.net.keep_prob: 1.})
        
        logging.info("Generator loss = {}, Discriminator loss = {}".format(loss_gen, loss_disc))
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
        
        os.mkdir(prediction_folder)
        utils.save_predictions_metric(batch_x, batch_y, prediction, masks, prediction_folder)
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_mask):
        # Calculate batch loss and accuracy
        summary_str, loss_gen, loss_disc = sess.run((self.summary_all, self.net.cost_generator, self.net.cost_discriminator),
                                                    feed_dict={self.net.x: batch_x, self.net.y: batch_y, self.net.mask: batch_mask, self.net.keep_prob: 1.})
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss: Generator = {:.16f}, Discriminator = {:.16f}".format(step, loss_gen, loss_disc))
        
    def output_epoch_stats(self, epoch, loss_gen, loss_disc, lr):
        logging.info(
            "Epoch {:}, Loss Generator = {:.16f}, Loss Discriminator = {:.16f}, learning rate: {:.8f}".format(epoch, loss_gen, loss_disc, lr))
    
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
            
            step_counter = 0
            for epoch in range(epochs):
                print(epoch)
                for step, (batch_x, batch_y, batch_mask) in enumerate(data_provider_train(self.batch_size)):
                    _, loss_gen, lr, gradients_gen = sess.run((self.optimizer_generator, self.net.cost_generator, self.learning_rate_node, self.net.gradients_node_generator),
                                                      feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.mask: batch_mask, self.net.keep_prob: keep_prob})
                    
                    _, loss_disc, lr, gradients_disc = sess.run((self.optimizer_discriminator, self.net.cost_discriminator, self.learning_rate_node, self.net.gradients_node_discriminator),
                                                      feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.mask: batch_mask, self.net.keep_prob: keep_prob})
                    
                    if self.net.create_summary and self.create_train_summary:
                        gradients_norm_ = [np.linalg.norm(gradient) for gradient in gradients_gen]
                        self.norm_gradients_node_gen.assign(gradients_norm_).eval()
                        
                        gradients_norm_ = [np.linalg.norm(gradient) for gradient in gradients_disc]
                        self.norm_gradients_node_disc.assign(gradients_norm_).eval()
                        
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, batch_y, batch_mask)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss_gen, loss_disc, lr)
                self.store_prediction(sess, test_x, test_y, masks, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
                
                if epoch % lr_update == 0 and epoch != 0:
                    sess.run(self.learning_rate_node.assign(self.learning_rate_node.eval()/2))
            
            summary_writer.close()
        logging.info("Training Finished")
        return save_path
