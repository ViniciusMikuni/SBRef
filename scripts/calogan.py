import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import h5py as h5
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import horovod.tensorflow.keras as hvd
import utils
import argparse
import tensorflow.keras.backend as K
from SBridge import SBridge
from WGAN import WGAN, discriminator_loss, generator_loss
from VAE import VAE
    

if __name__ == '__main__':
    #Start horovod and pin each GPU to a different process
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--nevts', default=-1,type=int, help='Number of events to load. -1 loads all')
    parser.add_argument('--file', default='/global/cfs/cdirs/m3929/SCRATCH/SGM/gamma.hdf5', help='File to load')

    parser.add_argument('--model', default='WGAN', help='Baseline ML model. Options: WGAN, VAE')
    parser.add_argument('--SB', action='store_true', default=False, help='Train the refinement model')
    parser.add_argument('--config', default='config_calogan.json', help='Config file with training parameters')


    parser.add_argument('--stage', default=0,type=int, help='IPF iteration to load')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    
    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)    
    
    nevts,tf_data1,energy = utils.CaloGAN_prep(flags.file,int(flags.nevts),use_logit=True)

    NUM_EPOCHS = 200
    LR = 1e-4*hvd.size()
    BATCH_SIZE=128

    if flags.model == 'WGAN' and not flags.SB:
        if hvd.rank()==0:print("Will train the base WGAN model")
        model = WGAN(ndim=504, num_cond=1)
        opt_gen = tf.optimizers.RMSprop(learning_rate=LR)
        opt_dis = tf.optimizers.RMSprop(learning_rate=0.1*LR)


        # Horovod: add Horovod DistributedOptimizer.
        opt_gen = hvd.DistributedOptimizer(
            opt_gen, backward_passes_per_step=1, average_aggregated_gradients=True)
        
        opt_dis = hvd.DistributedOptimizer(
            opt_dis, backward_passes_per_step=1, average_aggregated_gradients=True)

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
        
        if hvd.rank()==0:
            checkpoint = ModelCheckpoint('../checkpoint_WGAN/checkpoint',
                                         mode='auto',period=1,save_weights_only=True)
            callbacks.append(checkpoint)

    
        history = model.fit(
            tf_data1.batch(BATCH_SIZE),
            steps_per_epoch=nevts//BATCH_SIZE,
            # validation_data=samples_test.batch(BATCH_SIZE),
            # validation_steps=max(1,int(ntest/BATCH_SIZE)),
            epochs=NUM_EPOCHS,
            verbose=hvd.rank() == 0,
            callbacks=callbacks,
        )

        


    elif flags.model == 'VAE' and not flags.SB:
        if hvd.rank()==0:print("Will train the base VAE model")
        NUM_EPOCHS = 200
        
        LR = 1e-4*np.sqrt(hvd.size())
        #*hvd.size()
        model = VAE(ndim=504, num_cond=1)
        opt = tf.optimizers.Adam(learning_rate=LR)

        # Horovod: add Horovod DistributedOptimizer.
        opt = hvd.DistributedOptimizer(
            opt, backward_passes_per_step=1, average_aggregated_gradients=True)
        
        model.compile(optimizer=opt,experimental_run_tf_function=False,)        
    
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            # ReduceLROnPlateau(patience=10, min_lr=1e-7,verbose=hvd.rank() == 0),
            # EarlyStopping(patience=dataset_config['EARLYSTOP'],restore_best_weights=True),
        ]
        
        if hvd.rank()==0:
            checkpoint = ModelCheckpoint('../checkpoint_VAE/checkpoint',
                                         mode='auto',period=1,save_weights_only=True)
            callbacks.append(checkpoint)

    
        history = model.fit(
            tf_data1.batch(BATCH_SIZE),
            steps_per_epoch=nevts//BATCH_SIZE,
            # validation_data=samples_test.batch(BATCH_SIZE),
            # validation_steps=max(1,int(ntest/BATCH_SIZE)),
            epochs=NUM_EPOCHS,
            verbose=hvd.rank() == 0,
            callbacks=callbacks,
        )


    elif flags.SB:
        NUM_EPOCHS = config['EPOCH']
        NUM_STAGE=50
        LR = config['LR']
        BATCH_SIZE=config['BATCH']

        
        if hvd.rank()==0:print("Will train the refinement model based on the {} trained model".format(flags.model))
        if flags.model == 'WGAN':
            model = WGAN(ndim=504, num_cond=1)
            model.load_weights('{}/{}'.format('../checkpoint_WGAN','checkpoint')).expect_partial()
            generated = tf.data.Dataset.from_tensor_slices(model.generate(energy))
            tf_energy = tf.data.Dataset.from_tensor_slices(energy)
            tf_data2 = tf.data.Dataset.zip((generated,tf_energy)).shuffle(1000).repeat()

        elif flags.model == 'VAE':
            model = VAE(ndim=504, num_cond=1)
            model.load_weights('{}/{}'.format('../checkpoint_VAE','checkpoint')).expect_partial()
            generated = model.generate(energy).numpy()
            generated = np.clip(generated,-5,5)
            #generated = np.random.normal(size=(energy.shape[0],504)).astype(np.float32)
            
            generated = tf.data.Dataset.from_tensor_slices(generated)
            tf_energy = tf.data.Dataset.from_tensor_slices(energy)
            tf_data2 = tf.data.Dataset.zip((generated,tf_energy)).shuffle(1000).repeat()
        
                

        model = SBridge(name=flags.model,
                        #use_unet=True,
                        config=config,
                        use_conditional=True)


        if flags.load:
            assert flags.stage > 0
            cfs_folder = '/global/cfs/cdirs/m3929/SB'
            checkpoint_folder = '{}/checkpoints_{}_SB_Simple_{}'.format(cfs_folder,model.title,
                                                                        flags.stage-1)
            model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
        
        opt_b = tf.optimizers.Adam(learning_rate=LR)
        opt_f = tf.optimizers.Adam(learning_rate=LR)
        
        model.compile(opt_f=opt_f,opt_b=opt_b)
        
        model.fit_bridge(
            tf_data1.batch(BATCH_SIZE),
            tf_data2.batch(BATCH_SIZE),
            NUM_STAGE,
            NUM_EPOCHS,
            100000//(BATCH_SIZE*hvd.size()),
            start=flags.stage,
        )
