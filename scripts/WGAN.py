import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from architecture import get_unet, get_MLP


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)



class WGAN(keras.Model):
    """WGAN model"""
    def __init__(self, ndim,num_cond,num_noise=100,use_unet=False,):
        super(WGAN, self).__init__()
        self.num_cond = num_cond
        self.num_dim = ndim
        
        #config file with ML parameters to be used during training        
        #Transformation applied to conditional inputs
        inputs_cond = Input((self.num_cond))

        self.latent_dim = num_noise
        self.d_steps = 5 #number of discriminator steps for each generator
        self.gp_weight = 5.0 #weight of the gradient penalty to the loss
        self.num_embed=32

        if use_unet:
            inputs_g,outputs_g = get_unet(
                self.latent_dim,
                self.num_dim,
                layers.Reshape((1,time_embed.shape[-1]))(time_embed),
                image_embedding_dims = 32,
                block_depth = 1,
                widths = [32, 64],
                attentions = [False, False],
            )
            inputs_d,outputs_d =  get_unet(
                self.num_dim,
                1,
                layers.Reshape((1,time_embed.shape[-1]))(time_embed),
                image_embedding_dims = 32,
                block_depth = 1,
                widths = [32, 64],
                attentions = [False, False],
            )
            
        else:
            inputs_g,outputs_g = get_MLP(
                self.latent_dim,
                self.num_dim,
                inputs_cond,
                self.num_embed,
                mlp_dim=256,
                zeros=False,
                #use_clamp=True,
            )
            inputs_d,outputs_d = get_MLP(
                self.num_dim,
                1,
                inputs_cond,
                self.num_embed,
                mlp_dim=256,
                zeros=False,
            )


        self.discriminator = keras.models.Model([inputs_d,inputs_cond], outputs_d, name="discriminator")
        self.generator = keras.models.Model([inputs_g,inputs_cond], outputs_g, name="generator")


    def compile(self,d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile(run_eagerly=True )
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images,cond):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image

        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
            
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated,cond], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.

        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))        
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def generate(self,cond):
        random_latent_vectors = tf.random.normal(
            shape=(cond.shape[0], self.latent_dim)
        )
        return self.generator([random_latent_vectors,cond], training=False)
    
    def train_step(self, inputs):        
        real_images,cond = inputs
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                real_logits = self.discriminator([real_images,cond], training=True)
                fake_images = self.generator([random_latent_vectors,cond], training=True)
                fake_logits = self.discriminator([fake_images,cond], training=True)


                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images,cond)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            
            
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors,cond], training=True)
            gen_img_logits = self.discriminator([generated_images,cond], training=True)

            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)


        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )



        

        return {"d_loss": d_loss, "g_loss": g_loss}        
                        

    def test_step(self, inputs):
        real_images,cond = inputs
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        generated_images = self.generator([random_latent_vectors,cond], training=False)
        gen_img_logits = self.discriminator([generated_images,cond], training=False)
        real_logits = self.discriminator([real_images,cond], training=True)
        
        pred_energy_gen = self.energy_norm(generated_images, training=True)
        pred_energy_true = self.energy_norm(real_images, training=True)
        energy_loss = tf.reduce_mean(
                tf.abs(tf.square(pred_energy_gen-cond) - tf.square(pred_energy_true-cond)))
        
        gp = self.gradient_penalty(batch_size, real_images, generated_images,cond)
        d_loss = self.d_loss_fn(real_img=real_logits, fake_img=gen_img_logits)+ gp * self.gp_weight
        g_loss = self.g_loss_fn(gen_img_logits)
        
        return {"d_loss": d_loss, "g_loss": g_loss,'gp_loss':gp,'aux_loss':energy_loss}


    

if __name__ == '__main__':
    #Start horovod and pin each GPU to a different process
    hvd.init()    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='calogan', help='Name of the model to train. Options are: mnist, challenge, calogan')
    parser.add_argument('--nevts', default=-1, help='Name of the model to train. Options are: mnist, moon, calorimeter')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()
        
    model_name = flags.model #Let's try the parallel training using different models and different network architectures. Possible options are [mnist,moon,calorimeter]
        
    if model_name == 'calogan':        
        dataset_config = preprocessing.LoadJson('config_calogan_wgan.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test = preprocessing.CaloGAN_prep(file_path,int(flags.nevts),use_logit=True)
        use_1D = True
        
    elif model_name == 'calochallenge':
        dataset_config = preprocessing.LoadJson('config_challenge_wgan.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test = preprocessing.CaloChallenge_prep(file_path,int(flags.nevts),use_logit=False)
        use_1D = False
    else:
        raise ValueError("Model not implemented!")
        
        
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    BATCH_SIZE = dataset_config['BATCH']

    
    #Stack of bijectors
    model = WGAN(dataset_config['SHAPE'], 1, num_noise=dataset_config['NOISE_DIM'],config=dataset_config,use_1D=use_1D)

    opt_gen = tf.optimizers.Adam(learning_rate=LR)
    opt_dis = tf.optimizers.Adam(learning_rate=0.1*LR)


    # Horovod: add Horovod DistributedOptimizer.
    opt_gen = hvd.DistributedOptimizer(
        opt_gen, backward_passes_per_step=1, average_aggregated_gradients=True)

    opt_dis = hvd.DistributedOptimizer(
        opt_dis, backward_passes_per_step=1, average_aggregated_gradients=True)
    

    if flags.load:
        load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))


    model.compile(
        d_optimizer=opt_dis,
        g_optimizer=opt_gen,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )


    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        # ReduceLROnPlateau(patience=10, min_lr=1e-7,verbose=hvd.rank() == 0),
        # EarlyStopping(patience=dataset_config['EARLYSTOP'],restore_best_weights=True),
    ]

    
    history = model.fit(
        samples_train.batch(BATCH_SIZE),
        steps_per_epoch=int(ntrain/BATCH_SIZE),
        validation_data=samples_test.batch(BATCH_SIZE),
        validation_steps=max(1,int(ntest/BATCH_SIZE)),
        epochs=NUM_EPOCHS,
        verbose=hvd.rank() == 0,
        callbacks=callbacks,
    )

    if hvd.rank() == 0:            
        save_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
