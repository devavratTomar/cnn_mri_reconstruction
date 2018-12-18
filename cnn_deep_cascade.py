"""
Implementation of Deep Cascade in tensorflow
"""

import tensorflow as tf
import utils
import numpy as np
import logging
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_network(x, channels_x,
                        mask, cascade_n=5, layers=3, features=64, filter_size=3,
                        lambda_ = 3.,
                        reuse = False,
                        create_summary=True):
    """
    :param x: input_tensor, shape should be [None, n, m, channels_x]
    :param channels_x: number of channels in the input image. For Mri, input has 2 channels.
    :param mask: mask applied on fourier transform of input image. Shape must be [1, n, m]
    :param cascade_n: depth of cascade.
    :param layers: number of layers in deep cascade architecture.
    :param feature_base: Neurons in first layer of cnn. Next layers have twice the number of neurons in previous layers.
    :param filter_size: size of convolution filter
    :create_summary: Creates Tensorboard summary if True
    """
    
    logging.info("Layers: {layers}, features: {features}, filter size {fill_size}x{fill_size},"
                 "input channels {in_channels}".format(
                  layers=layers,
                  features=features,
                  fill_size=filter_size,
                  in_channels=channels_x))
    
    mask = tf.complex(mask, 0.)
    with tf.variable_scope("DeepCascade", reuse=reuse):
        #placeholder for input image
        with tf.name_scope("input_image"):
            n = tf.shape(x)[1]
            m = tf.shape(x)[2]
            
            x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
            x_image_cascade = x_image
            
        # create cascade layers
        for cascade in range(cascade_n):
            with tf.name_scope("cascade_net_{}".format(cascade)):
                with tf.variable_scope("cascade{}".format(cascade)):
                    # hidden layers
                    input_node = x_image_cascade
                    for layer in range(layers):
                        if layer == 0:
                            w = utils.weight_variable([filter_size, filter_size, channels_x, features], "w1_layer{}".format(layer))
                        else:
                            w = utils.weight_variable([filter_size, filter_size, features, features], "w1_layer{}".format(layer))
                        b = utils.bias_variable([features], "b_layer{}".format(layer))
                        conv = utils.conv2d(input_node, w, b, 1)
                        conv = tf.nn.leaky_relu(conv)        
                        
                        input_node = conv
                    
                    # aggregation layer
                    w = utils.weight_variable([filter_size, filter_size, features, channels_x], "w_agg")
                    b = utils.bias_variable([channels_x], "b_agg")
                    
                    output_image = utils.conv2d(input_node, w, b, 1)
                    
                    # don't apply activation when aggregating
                    # residual layer
                    output_image = tf.add(x_image_cascade, output_image)
            
                    #output image complex
                    output_image_complex = tf.complex(output_image[:, :, :, 0], output_image[:, :, :, 1])
                    output_image_complex_fft = tf.spectral.fft2d(output_image_complex)
            
                    #input image complex fft
                    input_image_complex = tf.complex(x_image[:, :, :, 0], x_image[:, :, :, 1])
                    input_image_complex_fft = tf.spectral.fft2d(input_image_complex)
            
                    #data consistency
                    output_image_complex_fft = tf.add((lambda_*input_image_complex_fft + tf.multiply(mask, output_image_complex_fft))/(lambda_ + 1.),
                                                      tf.multiply((1.0-mask), output_image_complex_fft))
                    
                    output_image_complex = tf.spectral.ifft2d(output_image_complex_fft)
                    output_image_complex = tf.reshape(output_image_complex, tf.stack([-1, n, m, 1]))
                    output_image_corrected = tf.concat([tf.real(output_image_complex), tf.imag(output_image_complex)],
                                                       axis=3)
                    
                    x_image_cascade = output_image_corrected
        
        return output_image_corrected


class DeepCascade(object):
    """
    Implementation of Deep Cascade.
    
    :param x_channels: number of channels in input image
    :param y_channels: number of channels in output image
    """
    
    def __init__(self, x_channels, layers, ncascade, mask_in, features, filter_size, create_summary=True):
        tf.reset_default_graph()
        
        self.x = tf.placeholder("float", shape=[None, None, None, x_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, x_channels], name="y")
        self.mask = mask_in.astype(np.float32)[np.newaxis, :, :]
        output_image = create_conv_network(x= self.x,
                                           channels_x= x_channels,
                                           mask=self.mask,
                                           cascade_n=ncascade,
                                           layers=layers,
                                           features=features,
                                           filter_size=filter_size,
                                           create_summary= create_summary)
        
        self.cost = self.__get_cost(output_image)        
        
        with tf.name_scope("resuts"):
            self.predictor = output_image
    
    def __get_cost(self, output_image):
        with tf.name_scope("cost"):
            loss = tf.losses.mean_squared_error(self.y, output_image)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="DeepCascade")
        return loss + 1e-6*sum(reg_losses)
    
    def predict(self, model_path, test_image):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((test_image.shape[0], test_image.shape[1], test_image.shape[2], test_image.shape[3]))
            prediction = sess.run(self.predictor, feed_dict={self.x: test_image, self.y: y_dummy})
            
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
    This will train a deep cascade instance.
    
    :param net: deep cascade to train
    :param batch_size: size of training batch
    :param validation_batch_size: size of validation batch
    :param create_train_summary: add training summaries if True (e.g. gradients)
    """
    
    def __init__(self, net, batch_size=1, validation_batch_size=1, create_train_summary=True):
        self.net = net
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.create_train_summary = create_train_summary
        
        # Tensorboard summary
        self.psnr     = tf.Variable(0., dtype="float")
        self.ssim     = tf.Variable(0., dtype="float")
        self.snr      = tf.Variable(0., dtype="float")
        self.l2_error = tf.Variable(0., dtype="float")
        self.l1_error = tf.Variable(0., dtype="float")
        
    def __get_optimizer(self, global_step):
        # we choose adam optimezer for this problem.
        learning_rate = 0.0001
        self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                            .minimize(self.net.cost, global_step=global_step)

        return optimizer
    
    def __initialize(self, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        
        self.optimizer = self.__get_optimizer(global_step)
        train_summary = [tf.summary.scalar("loss", self.net.cost),
                         tf.summary.scalar("learning_rate", self.learning_rate_node)]
        
        val_summary = [tf.summary.scalar("psnr", self.psnr),
                       tf.summary.scalar("ssim", self.ssim),
                       tf.summary.scalar("snr", self.snr),
                       tf.summary.scalar("l2_error", self.l2_error),
                       tf.summary.scalar("l1_error", self.l1_error)]
        
        self.summary_train = tf.summary.merge(train_summary)
        self.summary_val   = tf.summary.merge(val_summary)
        
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
    
    def store_prediction(self, sess, summary_writer, step, batch_x, batch_y, name):
        prediction, loss = sess.run((self.net.predictor, self.net.cost), feed_dict= {self.net.x: batch_x, self.net.y: batch_y})
        logging.info("Validaiton loss = {:.4f}".format(loss))
        
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
            
        os.mkdir(prediction_folder)
        metrics_val = utils.save_predictions_metric(batch_x, batch_y, prediction, self.net.mask[0], prediction_folder)
        sess.run((self.ssim.assign(metrics_val[0]),
                  self.snr.assign(metrics_val[1]),
                  self.psnr.assign(metrics_val[2]),
                  self.l2_error.assign(metrics_val[3]),
                  self.l1_error.assign(metrics_val[4])))
        
        summary_str = sess.run(self.summary_val)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss = sess.run((self.summary_train, self.net.cost),
                                     feed_dict={self.net.x: batch_x, self.net.y: batch_y})
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.16f}".format(step, loss))
        
        
    def output_epoch_stats(self, epoch, loss, lr):
        logging.info(
            "Epoch {:}, loss: {:.16f}, learning rate: {:.8f}".format(epoch, loss, lr))
    
    def train(self, data_provider_train, data_provider_validation,
              output_path,
              epochs=10,
              display_step=1,
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
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Training Started") 
            step_counter = 0
            
            test_x, test_y = data_provider_validation(self.validation_batch_size)
            self.store_prediction(sess, summary_writer, step_counter, test_x, test_y, "_init")
            
            for epoch in range(epochs):
                print(epoch)
                for step, (batch_x, batch_y) in enumerate(data_provider_train(self.batch_size)):
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                           feed_dict= {self.net.x: batch_x, self.net.y: batch_y})
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, batch_y)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss, lr)
                self.store_prediction(sess, summary_writer, step_counter, test_x, test_y, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
            
            summary_writer.close()
        logging.info("Training Finished")
        return save_path
