import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Input
from architecture import  get_encoder, get_decoder
# import tensorflow_probability as tfp
# from tensorflow_probability import distributions
import horovod.tensorflow.keras as hvd



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]

        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # z = distributions.MultivariateNormalDiag(
        #     loc = z_mean,scale_diag=tf.exp(z_log_var)).sample()        
        # return z

def soft_clamp(layer,n=5.0):
    return n*tf.math.tanh(layer/n)

class VAE(keras.Model):
    """VAE model"""
    def __init__(self, ndim,num_cond,num_noise=100,name='vae'):
        super(VAE, self).__init__()
        self.num_cond = num_cond
        self.num_dim = ndim
        self.num_embed = 64
        self.kl_steps=100*624//hvd.size()
        inputs_cond = Input((self.num_cond))
        self.latent_dim = num_noise


            

        inputs_e,z_mean, z_log_var, = get_encoder(
            self.num_dim,
            self.latent_dim,
            layers.Reshape((1,inputs_cond.shape[-1]))(inputs_cond),
            input_embedding_dims = 32,
            stride=2,
            kernel=3,
            block_depth = 1,
            widths = [16,32,64],
            attentions = [False, False, False],
        )
            
        z = Sampling()([z_mean, z_log_var])
            
            
        inputs_d,outputs_d = get_decoder(
            self.latent_dim,
            self.num_dim,
            inputs_cond,
            self.num_embed,
            num_layer=5,
            mlp_dim=512,
            activation='swish',
        )


        self.decoder = keras.models.Model([inputs_d,inputs_cond], outputs_d, name="decoder")
        self.encoder = keras.models.Model([inputs_e,inputs_cond], [z_mean, z_log_var, z], name="encoder")

        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="rec_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, inputs):
        data,cond = inputs
        with tf.GradientTape() as tape:            
            z_mean, z_log_var, z = self.encoder([data,cond])
            reconstruction = self.decoder([z,cond])
        
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data-reconstruction), axis=1)
            )
            
            #beta = tf.math.minimum(1.0,tf.cast(self.optimizer.iterations,tf.float32)/self.kl_steps)
            beta=1.0
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = beta*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, inputs):
        data,cond = inputs
        z_mean, z_log_var, z = self.encoder([data,cond])
        reconstruction = self.decoder([z,cond])
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data-reconstruction), axis=1)
        )
        beta = tf.math.minimum(1.0,tf.cast(self.optimizer.iterations,tf.float32)/self.kl_steps)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = beta*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    
    def generate(self,cond):
        random_latent_vectors = tf.random.normal(
            shape=(cond.shape[0], self.latent_dim)
        )
        return self.decoder([random_latent_vectors,cond], training=False)

