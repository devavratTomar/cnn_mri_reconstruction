"""
Implementation of Deep Cascade in tensorflow
"""

import tensorflow as tf
import utils
import numpy as np
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
IMAGE_SIZE = 256

def get_mask(mu, cov):
    # given parameters of probability distribution, sample mask
    m_index = np.random.multivariate_normal(mu, cov).astype(int)
    
    # clip out of bound index
    upper = m_index > IMAGE_SIZE - 1
    lower = m_index < 0
    
    m_index[upper] = int(IMAGE_SIZE - 1)
    m_index[lower] = 0
    
    mask = np.zeros([1, IMAGE_SIZE, IMAGE_SIZE])
    mask[0, m_index, :] = 1.0
    
    return mask, m_index

def mask_gradient_mu(loss, m_index, mu, cov):
    cov_inv = np.linalg.inv(cov)
    return loss*cov_inv.dot(m_index - mu)

def mask_gradient_cov(loss, m_index, mu, cov):
    cov_inv = np.linalg.inv(cov)
    m_index_diff = (m_index - mu)[:, np.newaxis]
    
    return -0.5*loss*cov_inv.dot(np.eye(cov.shape[0]) - (m_index_diff.dot(m_index_diff.T)).dot(cov_inv))

def create_conv_network(x, channels_x,
                        mask, cascade_n=5, layers=3, features=64, filter_size=3,
                        lambda_ = 3.,
                        reuse = False):
    """
    :param x: input_tensor, shape should be [None, n, m, channels_x]
    :param channels_x: number of channels in the input image. For Mri, input has 2 channels.
    :param mask: mask applied on fourier transform of input image. Shape must be [1, n, m]
    :param cascade_n: depth of cascade.
    :param layers: number of layers in deep cascade architecture.
    :param feature_base: Neurons in first layer of cnn. Next layers have twice the number of neurons in previous layers.
    :param filter_size: size of convolution filter
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
    :param layers: number of layers in each cascade
    :param ncascade: number of cascades
    :param mask_in: undersampling mask
    :param features: number of features in each hidden layers
    :param filter_size: size of the convolution filter
    """
    
    def __init__(self, x_channels, layers, ncascade, features, filter_size):
        tf.reset_default_graph()
        
        self.ground_truth = tf.placeholder("float", shape=[None, None, None, x_channels], name="ground_truth")
        self.mask = tf.placeholder("float", shape=[1, None, None], name="mask")
        
        ground_truth_complex = tf.complex(self.ground_truth[:, :, :, 0], self.ground_truth[:, :, :, 1])
        ground_truth_fft = tf.spectral.fft2d(ground_truth_complex)
        
        x_sub_complex = tf.reshape(tf.spectral.ifft2d(tf.multiply(ground_truth_fft, tf.complex(self.mask, 0.))), [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        
        self.x_sub = tf.concat([tf.real(x_sub_complex), tf.imag(x_sub_complex)], axis=3)
        
        output_image = create_conv_network(x= self.x_sub,
                                           channels_x= x_channels,
                                           mask=self.mask,
                                           cascade_n=ncascade,
                                           layers=layers,
                                           features=features,
                                           filter_size=filter_size)
        
        self.cost = self.__get_cost(output_image)
        
        with tf.name_scope("resuts"):
            self.predictor = output_image
    
    def __get_cost(self, output_image):
        with tf.name_scope("cost"):
            loss = tf.losses.mean_squared_error(self.ground_truth, output_image)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="DeepCascade")
        return loss + 1e-6*sum(reg_losses)
    
    def predict(self, model_path, test_image, mask):
        """
        Performs prediction on given test image and DeepCascade model. Returns fully sampled MRI prediction.
        
        :param model_path: path of the neural network model to load
        :param test_image: test images of shape [batch_size, None, None, 2]
        """
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            prediction = sess.run(self.predictor, feed_dict={self.ground_truth: test_image, self.mask: mask})
            
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
    """
    
    def __init__(self, net, batch_size=1, validation_batch_size=1):
        self.net = net
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        
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
        """
        Performs initialization of computation graph and tensorboard summary.
        
        :param output_path: The path where trained model will be saved at every checkpoint
        :param restore: If False, delete old path and create new model (should be used when training from scratch).
                        If True, we resotre the model. So don't delete the model at output_path
                        
        :param prediction_path: Path where prediction on test data will be saved after every epoch.
        """
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
    
    def store_prediction(self, sess, summary_writer, step, batch_x, mask, name):
        """
        Stores prediction images and metrics to tensorboard.
        """
        
        sub_sampled_image, prediction, loss = sess.run((self.net.x_sub, self.net.predictor, self.net.cost), feed_dict= {self.net.ground_truth: batch_x, self.net.mask: mask})
        logging.info("Validaiton loss = {:.4f}".format(loss))
        
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
            
        os.mkdir(prediction_folder)
        metrics_val = utils.save_predictions_metric(batch_x, sub_sampled_image, prediction, mask[0], prediction_folder)
        sess.run((self.ssim.assign(metrics_val[0]),
                  self.snr.assign(metrics_val[1]),
                  self.psnr.assign(metrics_val[2]),
                  self.l2_error.assign(metrics_val[3]),
                  self.l1_error.assign(metrics_val[4])))
        
        summary_str = sess.run(self.summary_val)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, mask):
        """
        Logs the all stats for current mini-batch to terminal and tensorboard.
        """
        
        # Calculate batch loss and accuracy
        summary_str, loss = sess.run((self.summary_train, self.net.cost),
                                     feed_dict={self.net.ground_truth: batch_x, self.net.mask: mask})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.16f}".format(step, loss))
        
        
    def output_epoch_stats(self, epoch, loss, lr):
        """
        Logs epoch stats to the terminal.
        """
        logging.info(
            "Epoch {:}, loss: {:.16f}, learning rate: {:.8f}".format(epoch, loss, lr))
    
    def train(self, data_provider_train, data_provider_validation,
              output_path,
              epochs=10,
              display_step=1,
              restore=False,
              write_graph=True,
              prediction_path='prediction',
              sampling_rate=0.25):
        """
        Start training the network
        
        :param data_provider_train: callback function returning training data
        :param data_provider_validation: callback function returning validation data
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
        
        # Initial mask is uniformly distributed with cov as diagnol
        mask_learning_rate = 1e-4
        
        k = int(sampling_rate*IMAGE_SIZE) - 1
        self.mu = np.arange(0, 255 + 255/k, 255/k)
        self.cov = 5.0*np.eye(k+1)
        
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
            
            test_x = data_provider_validation(self.validation_batch_size)
            self.store_prediction(sess, summary_writer, step_counter, test_x, get_mask(self.mu, self.cov)[0], "_init")
            
            for epoch in range(epochs):
                print(epoch)
                for step, batch_x in enumerate(data_provider_train(self.batch_size)):
                    mask, m_index = get_mask(self.mu, self.cov)
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                           feed_dict= {self.net.ground_truth: batch_x, self.net.mask: mask})
                    
                    # update mu and cov with respect to their gradients
                    # Ensure that cov is positive definite (Todo)
                    self.mu = self.mu - mask_learning_rate*mask_gradient_mu(loss, m_index, self.mu, self.cov)
                    self.cov = self.cov - mask_learning_rate*mask_gradient_cov(loss, m_index, self.mu, self.cov)
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, mask)
                        logging.info('mean:' self.mu)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss, lr)
                self.store_prediction(sess, summary_writer, step_counter, test_x, mask, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
            
            summary_writer.close()
            
        np.save('learned_mu', self.mu)
        np.save('learned_cov', self.cov)
        
        logging.info("Training Finished")
        return save_path
