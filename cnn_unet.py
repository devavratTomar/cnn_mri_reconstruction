"""
Implementation of Unet in tensorflow
"""

import tensorflow as tf
import utils
import numpy as np
import logging
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#TODO: Pass image size as parameter
IMAGE_SIZE = 128

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_resnet(x, channels_x, channels_y, layers, is_train, reuse=False):
    """
    Resnetwork as mention at https://arxiv.org/pdf/1609.04802.pdf
    """
    MINI_FILTER_SIZE = 3
    WIDE_FILTER_SIZE = 9
    
    logging.info("Layers: {layers}, input channels {in_channels}, output channels {out_channels}".format(\
                 layers=layers,
                 in_channels=channels_x,
                 out_channels=channels_y))
    
    with tf.variable_scope("Resnet", reuse=reuse):
        with tf.name_scope("Conv_init"):
            #placeholder for input image
            n = tf.shape(x)[1]
            m = tf.shape(x)[2]
            x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
            input_node = x_image
            
            # First layer is k9n64s1 (kernel size is 9x9, n_features is 64, stride is 1)
            w = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, channels_x, 64], name="weight_init")
            b = utils.bias_variable([64], name="bias_init")
            
            conv_init = utils.conv2d(input_node, w, b, stride=1)
            conv_init = tf.nn.leaky_relu(conv_init)
            
            input_node = conv_init
        
        for layer in range(layers):
            with tf.name_scope("Conv_layer{}".format(layer)):
                # Residual Blocks with format k3n64s1
                w = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 64, 64], name="weight_{}_1".format(layer))
                b = utils.bias_variable([64], "bias_{}_1".format(layer))
                
                conv = utils.conv2d(input_node, w, b, stride=1)
                bn = tf.layers.batch_normalization(conv, training=is_train)
                conv_activation = tf.nn.leaky_relu(bn)
                
                w = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 64, 64], name="weight_{}_2".format(layer))
                b = utils.bias_variable([64], "bias_{}_2".format(layer))
                
                conv = utils.conv2d(conv_activation, w, b, stride=1)
                bn = tf.layers.batch_normalization(conv, training=is_train)
                input_node = tf.add(bn, input_node)
                
        # Final layers
        with tf.name_scope("Conv_Final_1"):
            w = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 64, 64], name="weight_final_1")
            b = utils.bias_variable([64], name="bias_final_1")
            
            conv = utils.conv2d(input_node, w, b, stride=1)
            bn = tf.layers.batch_normalization(conv, training=is_train)
            
            input_node = tf.add(bn, conv_init)
            
        with tf.name_scope("Conv_Final_2"):
            w = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 64, 256], name="weight_final_2")
            b = utils.bias_variable([256], name="bias_final_2")
            
            conv = utils.conv2d(input_node, w, b, stride=1)
            conv = tf.nn.leaky_relu(conv)
            input_node = conv
            
        with tf.name_scope("Conv_Final_3"):
            w = utils.weight_variable([MINI_FILTER_SIZE, MINI_FILTER_SIZE, 256, 256], name="weight_final_3")
            b = utils.bias_variable([256], name="bias_final_3")
            
            conv = utils.conv2d(input_node, w, b, stride=1)
            conv = tf.nn.leaky_relu(conv)
            input_node = conv
            
        with tf.name_scope("Output_layer"):
            w = utils.weight_variable([WIDE_FILTER_SIZE, WIDE_FILTER_SIZE, 256, 2], name="weight_output")
            b = utils.bias_variable([2], name="bias_output")
            output_image = utils.conv2d(input_node, w, b, stride=1)
            
        output_image_complex = tf.complex(output_image[:, :, :, 0], output_image[:, :, :, 1])
        output_image_complex_fft = tf.spectral.fft2d(output_image_complex)
        output_image_complex_fft = tf.reshape(output_image_complex_fft, tf.stack([-1, n, m, 1]))
        output_image_fft = tf.concat([tf.real(output_image_complex_fft), tf.imag(output_image_complex_fft)], axis=3)
            
        return output_image, output_image_fft
            
class CnnResnet(object):
    """
    Implementation of ResnetClass.
    
    :param x_channels: number of channels in input image
    :param y_channels: number of channels in output image
    """
    
    def __init__(self, x_channels, y_channels, layers=3, create_summary=True):
        tf.reset_default_graph()
        
        self.x =    tf.placeholder("float", shape=[None, None, None, x_channels], name="x")
        self.y =    tf.placeholder("float", shape=[None, None, None, y_channels], name="y")
        self.mask = tf.placeholder("float", shape=[None, None, None, 1], name="mask")

        self.is_train = tf.placeholder("bool", name="batch_norm_is_train")
        
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.create_summary = create_summary
        
        output_image, output_image_fft = create_resnet(x= self.x,
                                                       channels_x= x_channels,
                                                       channels_y= y_channels,
                                                       layers= layers,
                                                       is_train=self.is_train)
        
        self.cost = self.__get_cost(output_image, output_image_fft)
        with tf.name_scope("resuts"):
            self.predictor = output_image
    
    def __get_cost(self, output, output_fft, lambda_=10):
        with tf.name_scope("cost"):
            loss_mse_image = tf.losses.mean_squared_error(self.y, output)
            
            input_image_cmplx = tf.complex(self.x[:, :, :, 0], self.x[:, :, :, 1])
            input_image_fft = tf.reshape(tf.spectral.fft2d(input_image_cmplx), tf.stack([-1, IMAGE_SIZE, IMAGE_SIZE, 1]))
            
            loss_mse_fft = tf.losses.mean_squared_error(tf.concat([tf.real(input_image_fft), tf.imag(input_image_fft)], axis=3),\
                                                        output_fft*self.mask)
        
        return lambda_*loss_mse_image + loss_mse_fft
    
    
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
        learning_rate = 0.0001
        self.learning_rate = tf.Variable(learning_rate, name="learning_rate")
    
    def __get_optimizer(self, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.cost, global_step=global_step)
            return optimizer
    
    def __initialize(self, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        
        tf.summary.scalar("loss", self.net.cost)
        self.optimizer = self.__get_optimizer(global_step)
        
        tf.summary.scalar("learning_rate", self.learning_rate)
        
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
                                          self.net.is_train: False})
        
        loss = sess.run(self.net.cost,
                        feed_dict= {self.net.x: batch_x,
                                    self.net.y: batch_y,
                                    self.net.mask: masks,
                                    self.net.is_train: False})
        
        logging.info("Validation Loss = {}".format(loss))
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
        
        os.mkdir(prediction_folder)
        utils.save_predictions_metric(batch_x, batch_y, prediction, masks, prediction_folder)
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_mask):
        # Calculate batch loss and accuracy
        summary_str, loss = \
        sess.run((self.summary_all, self.net.cost),
                 feed_dict={self.net.x: batch_x,
                            self.net.y: batch_y,
                            self.net.mask: batch_mask,
                            self.net.is_train: False})
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.16f}"\
                     .format(step, loss))
        
    def output_epoch_stats(self, epoch, loss, lr):
        logging.info(
            "Epoch {:}, Loss = {:.16f}, learning rate: {:.8f}".format(epoch, loss, lr))
    
    def train(self, data_provider_train, data_provider_validation,
              output_path,
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
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate),
                                           feed_dict= {self.net.x: batch_x,
                                                       self.net.y: batch_y,
                                                       self.net.mask: batch_mask,
                                                       self.net.is_train: True})
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, batch_y, batch_mask)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss, lr)
                self.store_prediction(sess, test_x, test_y, masks, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
                
                if epoch % lr_update == 0 and epoch != 0:
                    sess.run(self.learning_rate.assign(self.learning_rate.eval()/2.0))
            
            summary_writer.close()
        logging.info("Training Finished")
        return save_path
