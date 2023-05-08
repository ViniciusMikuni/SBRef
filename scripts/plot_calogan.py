import utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import horovod.tensorflow.keras as hvd
import pandas as pd
from sklearn.metrics import roc_curve, auc
from SBridge import SBridge
from WGAN import WGAN
from VAE import VAE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
utils.SetStyle()

hvd.init()


def AverageELayer(data_dict):
    
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        #preprocessed = np.sum(preprocessed,-1,keepdims=True)
        #preprocessed = np.mean(preprocessed,0)
        return preprocessed
        
    feed_dict = {}
    for key in data_dict:
        feed_dict[key] = _preprocess(data_dict[key])

    fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean deposited energy [GeV]')
    
    #ax0.set_yscale("log")
    fig.savefig('../plots/EnergyZ.pdf')
    return feed_dict


def HistEtot(data_dict):
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        return np.sum(preprocessed,-1)

    feed_dict = {}
    for key in data_dict:
        feed_dict[key] = _preprocess(data_dict[key])

            
    binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),1.3*np.quantile(feed_dict['Geant4'],1.0),20)
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning,reference_name='Geant4')
    ax0.set_xscale("log")
    fig.savefig('../plots/CaloGAN_TotalE.pdf')
    return feed_dict


def HistElayer(data_dict):
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        layer1=np.sum(preprocessed[:,:288],-1)
        layer2=np.sum(preprocessed[:,288:432],-1)
        layer3=np.sum(preprocessed[:,432:],-1)
        return layer1,layer2,layer3

    feed_dict = {}
    for ilayer in range(3):
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])[ilayer]

        print(np.min(feed_dict['Geant4']))
        binning = np.geomspace(max(1e-2,np.min(feed_dict['Geant4'])),1.2*np.max(feed_dict['Geant4']),15)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning,reference_name='Geant4')
        ax0.set_xscale("log")
        fig.savefig('../plots/CaloGAN_Layer{}E.pdf'.format(ilayer+1))
    return feed_dict
                    


def Classifier(data_dict,gen_name='WGAN'):
    from tensorflow import keras
    train = np.concatenate([data_dict['Geant4'],data_dict[gen_name]],0)
    labels = np.concatenate([np.zeros((data_dict['Geant4'].shape[0],1)),
                             np.ones((data_dict[gen_name].shape[0],1))],0)
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ])
    opt = tf.optimizers.Adam(learning_rate=2e-4)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.fit(train, labels,batch_size=1000, epochs=30)
    pred = model.predict(train)
    fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)    
    print("{} AUC: {}".format(auc(fpr, tpr),gen_name))
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='/global/cfs/cdirs/m3929/SCRATCH/SGM/gamma.hdf5', help='File to load')
    parser.add_argument('--config', default='config_calogan.json', help='Config file with training parameters')
    parser.add_argument('--model', default='WGAN', help='Baseline ML model. Options: WGAN, VAE')
    parser.add_argument('--nevts', default=-1,type=int, help='Number of events to load')
    parser.add_argument('--stage', default=3,type=int, help='IPF iteration to load')
    parser.add_argument('--cfs', default='/global/cfs/cdirs/m3929/SB', help='Folder where SB checkpoints are stored')
    
    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    
    use_logit=True
    energy, energy_voxel = utils.DataLoaderCaloGAN(flags.file,flags.nevts,use_logit=use_logit,use_noise=False)
    if flags.model == 'WGAN':
        model = WGAN(ndim=504, num_cond=1)
        model.load_weights('{}/{}'.format('../checkpoint_WGAN','checkpoint')).expect_partial()
        generated_voxel = model.generate(energy).numpy()
        del model
    elif flags.model == 'VAE':
        model = VAE(ndim=504, num_cond=1)
        model.load_weights('{}/{}'.format('../checkpoint_VAE','checkpoint')).expect_partial()
        generated_voxel = model.generate(energy).numpy()
        # generated_voxel = np.clip(generated_voxel,-5,5)
        del model
    

    model = SBridge(config=config,
                    use_conditional=True)
    checkpoint_folder = '{}/checkpoints_{}_SB_Simple_{}'.format(flags.cfs,flags.model,flags.stage)
    model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
    
    refined_voxel= []

    for split in np.array_split(np.column_stack((generated_voxel,energy)),10):        
        refined_voxel.append(model.propagate([split[:,:-1],split[:,-1]],model.ema_b,'backward')[0][:,-1].numpy())
    refined_voxel=np.concatenate(refined_voxel,0)

    energy_voxel = utils.ReverseNormCaloGAN(100*energy,energy_voxel,use_logit=use_logit)
    generated_voxel = utils.ReverseNormCaloGAN(100*energy,generated_voxel,use_logit=use_logit)
    refined_voxel = utils.ReverseNormCaloGAN(100*energy,refined_voxel,use_logit=use_logit)
                    
    data_dict = {
        flags.model:generated_voxel,
        'Geant4':energy_voxel,
        'refined':refined_voxel,
    }



    AverageELayer(data_dict)
    HistEtot(data_dict)
    HistElayer(data_dict)
    Classifier(data_dict,gen_name='refined')
    
