import json, yaml
import os
import h5py as h5
import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick



line_style = {
    'prior':'-',
    'posterior':'-',
    'combined':'-',
    'prior_forward':'dotted',
    'posterior_backward':'dotted',

    'SR':'-',
    'SB1':'-',
    'SB2':'-',
    'SB1_forward':'dotted',
    'SB2_backward':'dotted',

    'Geant4':'-',
    'WGAN':'dotted',
    'VAE':'dotted',
    'refined':'dotted',

}

colors = {
    'prior':'#7570b3',
    'posterior':'#d95f02',
    'prior_forward':'#1b9e77',
    'posterior_backward':'#e7298a',
    'combined':'b',

    'SR':'black',
    'SB1':'#7570b3',
    'SB2':'#d95f02',
    'SB1_forward':'#1b9e77',
    'SB2_backward':'#e7298a',

    'Geant4':'black',
    'WGAN':'#1b9e77',
    'VAE':'#1b9e77',
    'refined':'#e7298a',
}


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs





class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(-np.mean(feed_dict[reference_name],0)+np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-200,200])

    return fig,ax0


    
def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')

def GetEMD(ref,array):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(ref,array)
    # mse = np.square(ref-array)/ref
    # return np.sum(mse)


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='forward',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),20)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    for ip,plot in enumerate(feed_dict.keys()):
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step",weights=weights[plot])
        else:

            emdval = GetEMD(feed_dict[reference_name],feed_dict[plot])
            plot_label = r"{}, EMD :{:.2f}".format(plot,emdval)
                
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot_label
                              ,linestyle=line_style[plot],color=colors[plot],
                              density=True,histtype="step")
        
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to SR bkg.')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.7,1.3])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
          
    return fig,ax0




def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def ApplyPreprocessing(shower,fname='preprocessing.json',mask=None):
    params = LoadJson(fname)
    transform = (np.ma.divide((shower-params['mean']),params['std']).filled(0)).astype(np.float32)

    if mask is not None:
        return  transform*mask
    else:
        return  transform

def ReversePreprocessing(shower,fname='preprocessing.json',mask=None):
    params = LoadJson(fname)    
    transform = (params['std']*shower+params['mean']).astype(np.float32)
    
    if mask is not None:
        return  transform*mask
    else:
        return  transform


def CalcPreprocessing(shower,fname):
    mask = shower!=0
    mean = np.average(shower,axis=0)
    std = np.std(shower,axis=0)
    data_dict = {
        'mean':mean.tolist(),
        'std':np.std(shower,0).tolist(),
        'min':np.min(shower,0).tolist(),
        'max':np.max(shower,0).tolist()

    }

    
    SaveJson(fname,data_dict)
    print("done!")


def SmearVoxel(e,energy_voxel,factor=1.2):
    #smeared_voxel = np.random.normal(factor*energy_voxel,0.1*factor*np.abs(energy_voxel)+1e-5).astype(np.float32)
    smeared = energy_voxel.copy()
    smeared[np.squeeze(e)>0.5] = e[np.squeeze(e)>0.5]*energy_voxel[np.squeeze(e)>0.5]

    return smeared
    #sigmoid = 1.2/(1.0+np.exp(-e))
    return energy_voxel*sigmoid
    # return factor*energy_voxel
    # smeared_voxel = np.random.normal(size=energy_voxel.shape).astype(np.float32)
    return smeared_voxel

def ReverseNormCaloGAN(e,e_voxel,use_logit=True):
    '''Revert the transformations applied to the training set'''    
    alpha = 1e-6
    #gen_energy = 10**(e+1)
    #gen_energy = 100*e

    e_voxel = ReversePreprocessing(e_voxel,'preprocessing.json')
    if use_logit:
        # min_shower = np.ma.log(alpha/(1-alpha))
        # max_shower = np.ma.log((1-alpha)/alpha)
        # e_voxel=(e_voxel+1)/2.0        
        # e_voxel = min_shower + (max_shower-min_shower)*e_voxel
        exp = np.exp(e_voxel)
        x = exp/(1+exp)
        voxel = (x-alpha)/(1 - 2*alpha)
        voxel = voxel*e
        
    else:
        voxel = e_voxel*e 

    voxel[voxel<1e-4]=0.0
    return voxel


def DataLoaderCaloGAN(file_name,nevts=-1,use_logit=True,use_noise=True,
                      preprocessing_file='preprocessing_calogan.json'):
    '''
    Inputs:
    - name of the file to load
    - number of events to use
    Outputs:
    - Generated particle energy (value to condition the flow) (nevts,1)
    - Energy deposition in each layer (nevts,3)
    - Normalized energy deposition per voxel (nevts,504)
    '''
    import horovod.tensorflow as hvd
    #hvd.init()
    
    with h5.File(file_name,"r") as h5f:
        if nevts <0:
            nevts = len(h5f['energy'])
        e = h5f['energy'][hvd.rank():nevts:hvd.size()].astype(np.float32)

        layer0= h5f['layer_0'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0
        layer1= h5f['layer_1'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0
        layer2= h5f['layer_2'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0

    def preprocessing(data):
        ''' 
        Inputs: Energy depositions in a layer
        Outputs: Total energy of the layer and normalized energy deposition
        '''
        # x = data.shape[1]
        # y = data.shape[2]
        data_flat = np.reshape(data,[data.shape[0],-1])
        #add noise like caloflow does
        if use_noise:data_flat +=np.random.uniform(0,1e-7,size=data_flat.shape)        

        data_flat = np.ma.divide(data_flat,e).filled(0)
        #print(np.min(data_flat),np.max(data_flat))
        if use_logit:
            alpha = 1e-6
            #Log transform from caloflow paper
            
            x = alpha + (1 - 2*alpha)*data_flat            
            data_flat = np.ma.log(x/(1-x)).filled(0)
        else:
            pass
            #data_flat = 2*data_flat -1

                        
        return data_flat


    flat_shower = preprocessing(np.nan_to_num(layer0))    
    for il, layer in enumerate([layer1,layer2]):
        shower = preprocessing(np.nan_to_num(layer))
        flat_shower = np.concatenate((flat_shower,shower),-1)

    #CalcPreprocessing(flat_shower,'preprocessing.json')
    #input()
    flat_shower = ApplyPreprocessing(flat_shower,'preprocessing.json')

    flat_shower = np.clip(flat_shower,-5,5)
    return e/100.,flat_shower

def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    # print(tf.data.experimental.cardinality(test_data).numpy(),"cardinality")
    # input()
    return train_data,test_data


def CaloGAN_prep(file_path,nevts=-1,use_logit=False):    
    energy, energy_voxel = DataLoaderCaloGAN(file_path,nevts,use_logit=use_logit,use_noise=True)
    tf_energies = tf.data.Dataset.from_tensor_slices(energy)
    #tf_energies = tf.data.Dataset.from_tensor_slices(np.zeros(energy.shape).astype(np.float32))
    tf_data = tf.data.Dataset.from_tensor_slices(energy_voxel)
    data =tf.data.Dataset.zip((tf_data,tf_energies)).shuffle(energy.shape[0]).repeat()

    return energy.shape[0], data, energy



