#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:37:58 2018

@author: jd1336
"""

import numpy as np
import tensorflow as tf


class VariationalAutoencoder(object):
    """ Variational Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=64):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        
        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencoder network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
#        x_reconstr_mean = \
#            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
#                                 biases['out_mean']))

        x_reconstr_mean = \
            (tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        
        
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Gaussian distribution 
        #     induced by the decoder in the data space) is squared error
        reconstr_loss = tf.reduce_sum(tf.squared_difference(self.x, self.x_reconstr_mean), 1)
        # The loss is composed of two terms:
        # 2.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # reconstr_loss = \
        #    -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
        #                   + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
        #                   1)
        # 3.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost  = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost

    def calculate_testloss(self, X):
        test_cost  = self.sess.run(self.cost, feed_dict={self.x: X})
        return test_cost
    
    def transform(self, X): # maps to lower dimension
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def aggregate_posterior_z(self, X):
        
        mu_all, log_var_all = self.sess.run((self.z_mean,self.z_log_sigma_sq),
                                        feed_dict={self.x:X})
        var_all = np.exp(log_var_all)
        N = X.shape[0]        
        Exxt = 0        
        for i in range(N):
            mu_i  = mu_all[i,:]
            var_i = np.diag(var_all[i,:])
            Exxt =  Exxt + (var_i + mu_i.reshape((-1,1))*mu_i)
            
        mu_hat  = 1/N * np.sum(mu_all,0)
        var_hat =  1/N *Exxt - mu_hat.reshape((-1,1))*mu_hat        
        return mu_hat, var_hat
  
    def all_posterior_z(self, X):
        
        mu_all, log_var_all = self.sess.run((self.z_mean,self.z_log_sigma_sq),
                                        feed_dict={self.x:X})
        var_all = np.exp(log_var_all)
        
        return mu_all, var_all         
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})  
        
    def save(self, check_point_file = 'model.ckpt'):
        # Saves the weights (not the graph)
        save_path = self.saver.save(self.sess, check_point_file)
        print("saved the vae model weights to "+save_path)
    # to load it,
        
        
    def load(self, check_point_file = 'model.ckpt'):
        self.saver.restore(self.sess, check_point_file)
        print("loaded model weights from "+check_point_file)

    def calculate_testloss(self, X):
        cost  = self.sess.run(self.cost, 
                                  feed_dict={self.x: X})
        return cost
        
            
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)                
        
def train(vae, X_train, X_test, batch_size=64, training_epochs=10, display_step=5):
    n_samples       = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    total_batch = int(n_samples / batch_size)
    total_test_batch = int(n_test_samples / batch_size)
    inds_test = [i for i in range(n_test_samples)]
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        avg_test_cost = 0.0
        inds = np.random.permutation(n_samples)        
        # Loop over all batches
        for i in range(total_batch):            
            batch_xs = X_train[inds[i*batch_size:(i+1)*batch_size]]            
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            for ij in range(total_test_batch):
                batch_test_xs = X_test[inds_test[ij*batch_size:(ij+1)*batch_size]]
                cost_test  = vae.calculate_testloss(batch_test_xs)
                avg_test_cost += cost_test / n_test_samples * batch_size
            print("Epoch:", '%04d' % (epoch+1), 
                  "avg cost=", "{:.9f}".format(avg_cost))
            print("Epoch:", '%04d' % (epoch+1), 
                 "avg test cost=", "{:.9f}".format(avg_test_cost))
    return vae


