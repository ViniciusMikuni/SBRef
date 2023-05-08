import numpy as np
import os,re
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import time
import utils
import horovod.tensorflow as hvd
import argparse
import matplotlib.pyplot as plt
from architecture import get_unet, get_vit,get_MLP,get_conv
from tensorflow.keras.activations import swish
from tqdm import tqdm
import gc
utils.SetStyle()

tf.random.set_seed(12345)

class SBridge(keras.Model):
    """Schrodinger Bridge Tensorflow implementation"""
    def __init__(self, use_conditional=False,use_unet=False,
                 conditional_size=1,name='lhco',config=None):
        super(SBridge, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")
        
        self.num_dim=self.config['NDIM']
        self.num_embed=self.config['EMBED']
        self.num_steps =self.config['NSTEPS']
        self.use_conditional=use_conditional
        self.title=name
        self.activation = swish

        gamma_space = self.config['gamma_space']
        gamma_min = self.config['gamma_min']
        gamma_max = self.config['gamma_max']
        self.mean_match=False
        
        n = self.num_steps//2
        self.mean_final = 0.0
        self.var_final = 1.0*10**3 #parameters not really used in the case of non-gaussian bridges        

        if gamma_space == 'linspace':
            gamma_half = np.linspace(float(gamma_min), float(gamma_max), n)
        elif gamma_space == 'geomspace':
            gamma_half = np.geomspace(float(gamma_min), float(gamma_max), n)


        self.gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        self.T = tf.cast(tf.reduce_sum(self.gammas),tf.float32)
        self.time = np.cumsum(self.gammas, 0).astype(np.float32)/self.T
        self.T=1.0

        
        
        inputs_time = Input((1))
        self.projection = self.EmbeddingProjection(scale = 16)
        forward_conditional = self.Embedding(inputs_time,self.projection)
        backward_conditional = self.Embedding(inputs_time,self.projection)

        
        if use_conditional:
            inputs_cond = Input((conditional_size))
            cond_embed_forward = self.activation(layers.Dense(self.num_embed)(inputs_cond))
            cond_embed_backward = self.activation(layers.Dense(self.num_embed)(inputs_cond))
            forward_conditional = tf.concat([forward_conditional,cond_embed_forward],-1)
            backward_conditional = tf.concat([backward_conditional,cond_embed_backward],-1)


        if use_unet:

            inputs_f,outputs_f = get_conv(
                self.num_dim,
                layers.Reshape((1,forward_conditional.shape[-1]))(forward_conditional),
                conv_dim=[16,32],
            )
            inputs_b,outputs_b =  get_conv(
                self.num_dim,
                layers.Reshape((1,backward_conditional.shape[-1]))(backward_conditional),
                conv_dim=[16,32],
            )            
        else:
            inputs_f,outputs_f = get_MLP(
                self.num_dim,
                self.num_dim,
                forward_conditional,
                self.num_embed,
                num_layer=self.config['NUM_LAYER'],
                mlp_dim=self.config['MLP_DIM'],
            )
            inputs_b,outputs_b = get_MLP(
                self.num_dim,
                self.num_dim,
                backward_conditional,
                self.num_embed,
                num_layer=self.config['NUM_LAYER'],
                mlp_dim=self.config['MLP_DIM'],

            )
            

        if use_conditional:
            self.forward = keras.models.Model([inputs_f,inputs_time,inputs_cond], outputs_f, name="forward")
            self.backward = keras.models.Model([inputs_b,inputs_time,inputs_cond], outputs_b, name="backward")
        else:
            self.forward = keras.models.Model([inputs_f,inputs_time], outputs_f, name="forward")
            self.backward = keras.models.Model([inputs_b,inputs_time], outputs_b, name="backward")
        
        self.ema_f = keras.models.clone_model(self.forward)
        self.ema_b = keras.models.clone_model(self.backward)
        


    def EmbeddingProjection(self,scale = 30):
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding


    def compile(self,opt_f, opt_b):
        super(SBridge, self).compile(experimental_run_tf_function=False,
                                        #run_eagerly=True
        )
        self.opt_f = opt_f
        self.opt_b = opt_b

    

    def propagate(self,sample,model,direction='forward'):

        
        if self.use_conditional:
            x,cond = sample
        else:
            x = sample

        N = tf.shape(x)[0]
        #time = self.time if direction=='forward' else np.flip(self.time,0)
        time=self.time
        time = tf.cast(tf.repeat(tf.reshape(time,(1, self.num_steps, 1)),N,0),tf.float32)
        gammas = tf.cast(tf.repeat(tf.reshape(self.gammas,(1, self.num_steps, 1)),N,0),tf.float32)
 
        for idx in range(self.num_steps):
            gamma = self.gammas[idx]
            if self.use_conditional:
                t_old = model([x,time[:,idx],cond])
            else:
                t_old = model([x,time[:,idx]])
            if not self.mean_match:
                t_old += x

            if (idx == self.num_steps-1):
                x = t_old
            else:
                z = tf.random.normal(tf.shape(x))            
                x = t_old + tf.cast(tf.sqrt(2 * gamma),tf.float32)*z

            if self.use_conditional:
                t_new = model([x,time[:,idx],cond])
            else:
                t_new = model([x,time[:,idx]])

            if not self.mean_match:
                t_new += x
                
            if idx ==0:
                xs = tf.expand_dims(x,1)
                out = tf.expand_dims(t_old - t_new,1)
            else:
                xs = tf.concat([xs,tf.expand_dims(x,1)],1)
                out = tf.concat([out,tf.expand_dims(t_old - t_new,1)],1)

        return tf.convert_to_tensor(xs,dtype=tf.float32),tf.convert_to_tensor(out,dtype=tf.float32), time



    def propagate_first(self,sample,direction='forward'):
        #First difufsion without any learning model
        if self.use_conditional:
            x,cond = sample
        else:
            x = sample

        N = tf.shape(x)[0]
        time = tf.cast(tf.repeat(tf.reshape(self.time,(1, self.num_steps, 1)),N,0),tf.float32)
        gammas = tf.cast(tf.repeat(tf.reshape(self.gammas,(1, self.num_steps, 1)),N,0),tf.float32)
 

            
        for idx in range(self.num_steps):

            def grad_gauss(x,mean,var):
                return -(x-mean)/var

            
            gamma = self.gammas[idx]
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_old = x + gamma * gradx
            
            z = tf.random.normal(tf.shape(x))
            x = t_old + tf.cast(tf.sqrt(2 * gamma),tf.float32)*z
            
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_new = x + gamma * gradx

            if idx ==0:
                xs = tf.expand_dims(x,1)
                out = tf.expand_dims(t_old - t_new,1)
            else:
                xs = tf.concat([xs,tf.expand_dims(x,1)],1)
                out = tf.concat([out,tf.expand_dims(t_old - t_new,1)],1)

        return tf.convert_to_tensor(xs,dtype=tf.float32),tf.convert_to_tensor(out,dtype=tf.float32), time

    
    def get_loss_fn(self):
        @tf.function
        def loss_fn(sample,policy_opt, policy_impt,
                    sample_direction,opt,
                    ema_opt,
                    stage=1,first_epoch=False):
        
            if stage==0 and sample_direction == 'backward':        
                #training forward sampling from backward
                train_xs,train_zs,steps = self.propagate_first(sample,sample_direction)
            else:
                train_xs,train_zs,steps = self.propagate(sample,policy_impt,sample_direction)
                
            
            with tf.GradientTape() as tape:
                # prepare training data
                
                # -------- handle for batch_x and batch_t ---------
                # (batch, T, xdim) --> (batch*T, xdim)
                xs      = tf.reshape(train_xs,(-1,self.num_dim)) 
                zs_impt = tf.reshape(train_zs,(-1,self.num_dim))
                ts = tf.reshape(self.T -tf.cast(steps,dtype=tf.float32),(-1,1))

                

                # -------- compute loss and backprop --------
                #Estimate divergent using gaussian noise
                if self.use_conditional:
                    data,cond = sample
                    num_cond = tf.shape(cond)[-1]
                    cond = tf.expand_dims(cond,-2)
                    cond = tf.repeat(cond,self.num_steps,-2)
                    cond = tf.reshape(cond,(-1,num_cond))
                    pred = policy_opt([xs, ts,cond])
                    
                else:
                    pred = policy_opt([xs, ts])

                if self.mean_match:
                    pred -= xs
                        
                loss = tf.square(pred - zs_impt)
                loss = tf.reduce_mean(loss)

            tape = hvd.DistributedGradientTape(tape)
            variables = policy_opt.trainable_variables
            grads = tape.gradient(loss, variables)
            grads = [tf.clip_by_norm(grad, 1)
                     for grad in grads]
            opt.apply_gradients(zip(grads, variables))
            
            ema=0.999
            for weight, ema_weight in zip(policy_opt.weights, ema_opt.weights):
                ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
            
            if first_epoch:
                hvd.broadcast_variables(policy_opt.variables, root_rank=0)
                hvd.broadcast_variables(opt.variables(), root_rank=0)
                    
                
            return loss
        return loss_fn


    def reset_opt(self,opt):
        for var in opt.variables():
            var.assign(tf.zeros_like(var))
        return tf.constant(10)


    def sb_alternate_train_stage(self,iterator,stage, epoch, direction,loss_fn,NUM_STEPS):
        policy_opt, policy_impt, sample_direction,opt,ema_opt = {
            'forward':  [self.forward, self.backward,'backward',self.opt_f,self.ema_f], # train forwad,   sample from backward
            'backward': [self.backward, self.forward,'forward',self.opt_b,self.ema_b], # train backward, sample from forward
        }.get(direction)

        if hvd.rank()==0:
            epochs = tqdm(range(epoch))
        else:
            epochs = range(epoch)

        patience = 0
        stop_ep=0
        early_stopping=10
        min_epoch = np.inf
        for ep in epochs:
            sum_loss = []
            #print('Training epoch {}'.format(ep))
            for step in range(NUM_STEPS):
                first_epoch = stage==self.start and step==0 and ep ==0
                sample = iterator.get_next()
                loss = loss_fn(sample,
                               policy_opt, policy_impt,
                               sample_direction,opt,
                               ema_opt,
                               stage,first_epoch=first_epoch)
                sum_loss.append(loss.numpy())
            if np.mean(sum_loss) < min_epoch:
                min_epoch=np.mean(sum_loss)
                patience=0
            else:
                patience+=1
            if patience >=early_stopping and stop_ep==0:
                stop_ep = ep
            if hvd.rank()==0:
               epochs.set_description("Loss: {}".format(loss))
            gc.collect()
        return tf.reduce_mean(sum_loss),stop_ep


    def fit_bridge(self,
                   prior,
                   posterior,
                   NUM_STAGE,
                   NUM_EPOCHS,
                   NUM_STEPS,
                   start=0,
                   
    ):        
        iterator_prior = iter(prior)
        iterator_posterior = iter(posterior)
        loss_fn_f = self.get_loss_fn()
        loss_fn_b = self.get_loss_fn()
        self.start=start
        for stage in range(start,NUM_STAGE):
            if hvd.rank()==0:print("Training stage {}".format(stage))
            # Note: first stage of forward policy must converge;
            # otherwise it will mislead backward policy
            forward_ep = 300 if stage ==0 else NUM_EPOCHS
            backward_ep = 200 if stage ==0 else NUM_EPOCHS

            # train forward policy
            loss,ep = self.sb_alternate_train_stage(iterator_posterior,stage, forward_ep, 'forward',loss_fn_f,NUM_STEPS)
            if hvd.rank()==0:print("Trained forward model with loss {} in {} epochs".format(loss,ep))
            # train backward policy;            
            gc.collect()
            
            loss,ep = self.sb_alternate_train_stage(iterator_prior,stage, backward_ep, 'backward',loss_fn_b,NUM_STEPS)
            if hvd.rank()==0:
                print("Trained backward model with loss {} in {} epochs".format(loss,ep))
            gc.collect()
                                    
            if hvd.rank()==0:
                cfs_folder = '/global/cfs/cdirs/m3929/SB'
                checkpoint_folder = '{}/checkpoints_{}_SB_Simple_{}'.format(cfs_folder,self.title,stage)
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                self.save_weights('{}/{}'.format(checkpoint_folder,'checkpoint'),save_format='tf')
    
if __name__ == '__main__':
    pass
