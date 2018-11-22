"""
Implementation of Unet in tensorflow
"""

import tensorflow as tf
import utils
import numpy as np
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_network(x, channels_x, channels_y, layers=3, feature_base=16, feature_reconstruction=16, keep_prob=0.8, create_summary=True):
    """
    :param x: input_tensor, shape should be [None, n, m, channels_x]
    :param channels_x: number of channels in the input image. For Mri, input has 2 channels.
    :param channels_y: number of channels in the output image. For Mri, output has 2 channels.
    :param layers: number of layers in u-net architecture.
    :param feature_base: Neurons in first layer of cnn. Next layers have twice the number of neurons in previous layers.
    :create_summary: Creates Tensorboard summary if True
    """
    FILTER_SIZE_FEATURES = 3
    FILTER_SIZE_RECONSTRUCTION = 1
    RECONSTRUCTION_LAYERS = 2
    
    logging.info("Layers: {layers}, features: {features}, input channels {in_channels}, output channels {out_channels}".format(
                  layers=layers,
                  features=feature_base,
                  in_channels=channels_x,
                  out_channels=channels_y))
    
    #placeholder for input image
    with tf.name_scope("input_image"):
        n = tf.shape(x)[1]
        m = tf.shape(x)[2]
        
        x_image = tf.reshape(x, tf.stack([-1, n, m, channels_x]))
        input_node = x_image
        
    weights = []
    biases = []
    convs = []
    feature_concat = []
    
    # feature extraction layers
    for layer in range(layers):
        with tf.name_scope("feature_extraction_layer{}".format(str(layer))):
            std_dev = np.sqrt(2./(FILTER_SIZE_FEATURES*FILTER_SIZE_FEATURES*feature_base))
            
            if  layer == 0:
                weight = utils.weight_variable([FILTER_SIZE_FEATURES, FILTER_SIZE_FEATURES, channels_x, feature_base], std_dev, "w1")
            else:
                weight = utils.weight_variable([FILTER_SIZE_FEATURES, FILTER_SIZE_FEATURES, feature_base, feature_base], std_dev, "w1")   
            
            bias = utils.bias_variable([feature_base], "b1")
            conv_layer = utils.conv2d(input_node, weight, bias, keep_prob)
            conv_layer_activation = tf.nn.relu(conv_layer)
            
            feature_concat.append(conv_layer_activation)
            
            weights.append(weight)
            biases.append(bias)
            convs.append(conv_layer)        
            input_node = conv_layer_activation
    
    # collect all features in a single stack
    features_concat_all = tf.concat(feature_concat, 3)
    input_node = features_concat_all
    size_all_features = len(feature_concat)*feature_base
    
    # reconstruction layers
    for layer in range(RECONSTRUCTION_LAYERS):
        with tf.name_scope("reconstruction_layer{}".format(str(layer))):
            
            if layer == 0:
                std_dev = np.sqrt(2./(FILTER_SIZE_RECONSTRUCTION*FILTER_SIZE_RECONSTRUCTION*feature_reconstruction))
                weight = utils.weight_variable([FILTER_SIZE_RECONSTRUCTION, FILTER_SIZE_RECONSTRUCTION, size_all_features, feature_reconstruction], std_dev, "w2")
                
            else:
                std_dev = np.sqrt(2./(FILTER_SIZE_FEATURES*FILTER_SIZE_FEATURES*feature_reconstruction))
                weight = utils.weight_variable([FILTER_SIZE_FEATURES, FILTER_SIZE_FEATURES, feature_reconstruction, feature_reconstruction], std_dev, "w2")
            
            bias = utils.bias_variable([feature_reconstruction], "b1")
            conv_layer = utils.conv2d(input_node, weight, bias, keep_prob)
            conv_layer_activation = tf.nn.relu(conv_layer)
            
            weights.append(weight)
            biases.append(bias)
            convs.append(conv_layer)
            
            input_node = conv_layer_activation
    
    # bypass reconstruction layer
    with tf.name_scope("bypass_reconstruction_layer"):
        std_dev = np.sqrt(2./(FILTER_SIZE_RECONSTRUCTION*FILTER_SIZE_RECONSTRUCTION*feature_reconstruction))
        weight = utils.weight_variable([FILTER_SIZE_RECONSTRUCTION, FILTER_SIZE_RECONSTRUCTION, size_all_features, feature_reconstruction], std_dev, "w_bypass")
        bias = utils.bias_variable([feature_reconstruction], "b_bypass")
        conv_layer = utils.conv2d(features_concat_all, weight, bias, keep_prob)
        conv_layer_activation = tf.nn.relu(conv_layer)
        
        weights.append(weight)
        biases.append(bias)
        convs.append(conv_layer)
        
        input_node = tf.concat([conv_layer_activation, input_node], 3)
        
    # final reconstruction layer
    size_final_input = 2*feature_reconstruction
    
    std_dev = np.sqrt(2./(FILTER_SIZE_RECONSTRUCTION*FILTER_SIZE_RECONSTRUCTION*channels_y))
    weight = utils.weight_variable([FILTER_SIZE_RECONSTRUCTION, FILTER_SIZE_RECONSTRUCTION, size_final_input, channels_y], std_dev, "out_weight")
    bias = utils.bias_variable([channels_y], "out_weight")
    output_image = utils.conv2d(input_node, weight, bias, tf.constant(1.0))
    
    weights.append(weight)
    biases.append(bias)  
    
    # Create Summaries
    if create_summary:
        with tf.name_scope("summaries"):
            for i, conv in enumerate(convs):
                tf.summary.image("summary_conv_{:02}".format(i), utils.get_image_summary(conv))
    
    variables = weights + biases
    return output_image, variables 


class CnnUnet(object):
    """
    Implementation of Unet.
    
    :param x_channels: number of channels in input image
    :param y_channels: number of channels in output image
    """
    
    def __init__(self, x_channels, y_channels, layers, feature_base, feature_reconstruction, create_summary=True):
        tf.reset_default_graph()
        
        self.x = tf.placeholder("float", shape=[None, None, None, x_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, y_channels], name="y")
        self.keep_prob = tf.placeholder("float", name="dropout_probability")
        
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.create_summary = create_summary
        
        output_image, self.variables = create_conv_network(x= self.x,
                                                           channels_x= x_channels,
                                                           channels_y= y_channels,
                                                           layers= layers,
                                                           feature_base= feature_base,
                                                           feature_reconstruction= feature_reconstruction,
                                                           keep_prob=self.keep_prob,
                                                           create_summary= create_summary)
        
        self.cost = self.__get_cost(output_image)
        self.gradients_node = tf.gradients(self.cost, self.variables)
        
        with tf.name_scope("resuts"):
            self.predictor = output_image
    
    def __get_cost(self, output_image, regulizer=0):
        with tf.name_scope("cost"):
            loss = tf.losses.mean_squared_error(self.y, output_image, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            if regulizer !=0:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += regulizer*regularizers
        return loss
    
    def predict(self, model_path, test_image):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((test_image.shape[0], test_image.shape[1], test_image.shape[2], self.y_channels))
            prediction = sess.run(self.predictor, feed_dict={self.x: test_image, self.y: y_dummy, self.keep_prob: 1.0})
            
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
    
    def __get_optimizer(self, global_step):
        # we choose adam optimezer for this problem.
        learning_rate = 0.001
        self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                            .minimize(self.net.cost, global_step=global_step)

        return optimizer
    
    def __initialize(self, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradient")
        
        if self.net.create_summary and self.create_train_summary:
            tf.summary.histogram("norm_gradients", self.norm_gradients_node)
            
        tf.summary.scalar("loss", self.net.cost)
        
        self.optimizer = self.__get_optimizer(global_step)
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
        prediction = sess.run(self.net.predictor, feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.})
        loss = sess.run(self.net.cost, feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.})
        
        logging.info("Validaiton loss = {:.4f}".format(loss))
        
        prediction_folder = os.path.join(self.prediction_path, name)
        
        if os.path.exists(prediction_folder):
            shutil.rmtree(prediction_folder, ignore_errors=True)
            
        os.mkdir(prediction_folder)
        utils.save_predictions(batch_x, batch_y, prediction, masks, prediction_folder)
        
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, predictions = sess.run((self.summary_all, self.net.cost, self.net.predictor),
                                                  feed_dict={self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.})
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.16f}".format(step, loss))
        
        
    def output_epoch_stats(self, epoch, loss, lr):
        logging.info(
            "Epoch {:}, loss: {:.16f}, learning rate: {:.8f}".format(epoch, loss, lr))

    
    def train(self, data_provider_train, data_provider_validation,
              output_path,
              keep_prob,
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
                    
            test_x, test_y, masks = data_provider_validation(self.validation_batch_size)
            self.store_prediction(sess, test_x, test_y, masks, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Training Started")
            
            step_counter = 0
            for epoch in range(epochs):
                print(epoch)
                for step, (batch_x, batch_y) in enumerate(data_provider_train(self.batch_size)):
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                                                      feed_dict= {self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: keep_prob})
                    
                    if self.net.create_summary and self.create_train_summary:
                        gradients_norm = [np.linalg.norm(gradient) for gradient in gradients]
                        self.norm_gradients_node.assign(gradients_norm).eval()
                        
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step_counter, batch_x, batch_y)
                    step_counter +=1
                        
                self.output_epoch_stats(epoch, loss, lr)
                self.store_prediction(sess, test_x, test_y, masks, "epoch_{}".format(epoch))
                save_path = self.net.save(sess, save_path)
            
            summary_writer.close()
        logging.info("Training Finished")
        return save_path