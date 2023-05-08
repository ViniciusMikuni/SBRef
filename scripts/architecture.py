import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_addons as tfa

def soft_clamp(layer,n=5.0):
    return n*tf.math.tanh(layer/n)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size,is_1D=False, **kwargs):
        super().__init__(**kwargs)
        self.is_1D = is_1D
        self.embed_dim = embed_dim
        self.patch_size=patch_size
        
        if is_1D:
            self.projection = layers.Conv1D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
                use_bias=False,
            )
        else:
            self.projection = layers.Conv3D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
                use_bias=False,
            )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "is_1D": self.is_1D,
            "embed_dim": self.embed_dim,
            "patch_size":self.patch_size,
        })
        return config
    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config

def time_dense(input_layer,embed,hidden_size,activation=True):
    #Incorporate the time information to each layer used in the model
    layer = tf.concat([input_layer,embed],-1)
    layer = layers.Dense(hidden_size,activation=None)(layer)
    
    #layer = layers.LayerNormalization(epsilon=1e-6)(layer)
    # layer = layers.Dropout(0.1)(layer)
    if activation:            
        return keras.activations.swish(layer)
    else:
        return layer




def get_conv(
        num_dim,
        time_embedding,
        conv_dim=[16,32,32],
        input_embedding_dims=16,
):

    inputs = keras.Input((num_dim))
    inputs_expanded = layers.Reshape((num_dim,1))(inputs)    
    x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs_expanded)
    n = layers.UpSampling1D(size=num_dim)(time_embedding)
    x = layers.Concatenate()([x, n])

        
    for width in conv_dim:
        x = ResidualBlock(width,kernel=3,attention=False)(x)

    x = layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)
    outputs = tf.reshape(x,(-1,num_dim))
    return inputs,outputs

def get_vit(
        num_dim,
        end_dim,
        time_embedding,
        transformer_layers=1,
        num_heads=1,
        projection_dim=128,
        mlp_dim=1024,
        patch_size=12,
        use_clamp=False,
):
    
    ''' Visual transformer model as the network backbone for the FFJORD implementation'''
    inputs = keras.Input((num_dim))
    inputs_expanded = layers.Reshape((num_dim,1))(inputs)

    #residual = layers.Conv1D(projection_dim, kernel_size=1)(inputs_expanded)
    
    time_embed = layers.Dense(projection_dim)(time_embedding)
    time_embed = layers.Reshape((1,time_embed.shape[-1]))(time_embed)
    
    #time_layer = tf.tile(time_embed,[1,num_dim,1])                    
    #inputs_reshape = tf.concat([inputs_expanded,time_layer],-1)
    
    patches = TubeletEmbedding(embed_dim=projection_dim,is_1D=True,
                               patch_size=patch_size)(inputs_expanded)
        
    # Encode patches.
    encoded_patches = PositionalEncoder(embed_dim=projection_dim)(patches)


        
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        #x3=x2
        # MLP.
        #tf.nn.gelu

        # x3 = self.time_dense(x3,time_patch,2*projection_dim)
        # x3 = self.time_dense(x3,time_patch,projection_dim)
        
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    #pooling = layers.GlobalAvgPool1D()(representation)
    representation = layers.Flatten()(representation)
    #representation = time_dense(tf.concat([representation,pooling],-1),time_embedding,mlp_dim)
    representation = time_dense(representation,time_embedding,mlp_dim)
    #representation = layers.Dropout(0.1)(representation)
    representation = time_dense(representation,time_embedding,mlp_dim//2)
    #representation = layers.Dense(mlp_dim//2,activation='tanh')(representation)
    outputs = layers.Dense(end_dim)(representation)
    
    # representation = layers.UpSampling1D(size=patch_size)(representation)
    # representation = layers.Add()([residual, representation])    
    # layer = tf.concat([representation,time_layer],-1)        
    # layer = layers.Conv1D(projection_dim,activation="swish", kernel_size=1)(layer)    
    # layer = layers.Conv1D(1,activation=None, kernel_size=1)(layer)
    
    # outputs = layers.Flatten()(layer)

    
    # representation = layers.Flatten()(representation)
    # time_embed=layers.Flatten()(time_embedding)
    # layer = tf.concat([representation,time_embed],-1)
    # layer = layers.Dense(mlp_dim,activation="swish")(layer)
    # #layer = layers.Dense(mlp_dim//2,activation="swish")(layer)    
    # outputs = layers.Dense(end_dim)(layer)
    
    # if use_clamp:
    #     outputs = soft_clamp(outputs)

    
    
    return inputs, outputs


def ResidualBlock(width, kernel,attention):
    def forward(x):
        input_width = x.shape[2]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv1D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv1D(
            width, kernel_size=kernel, padding="same", activation=keras.activations.swish
        )(x)
        if attention:
            x = layers.MultiHeadAttention(
                num_heads=1, key_dim=width, attention_axes=(1))(x, x)
        else:
            x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
        x = layers.Add()([residual, x])
        return x

    return forward

def DownBlock(block_depth, width, stride,kernel,attention,use_skip=False):
    def forward(x):
        if use_skip:
            x, skips = x
            
        for _ in range(block_depth):
            x = ResidualBlock(width,kernel, attention)(x)
            if use_skip: skips.append(x)
        x = layers.AveragePooling1D(pool_size=stride)(x)
        return x

    return forward

def UpBlock(block_depth, width, stride,kernel,attention,use_skip=False):
    def forward(x):
        if use_skip:
            x, skips = x
        x = layers.UpSampling1D(size=stride)(x)
        for _ in range(block_depth):
            if use_skip: x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width,kernel, attention)(x)
        return x

    return forward




def get_unet(
        num_dim,
        end_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        use_clamp=False,
):
    #https://github.com/beresandras/clear-diffusion-keras/blob/master/architecture.py
    
    inputs = keras.Input((num_dim))
    inputs_expanded = layers.Reshape((num_dim,1))(inputs)    
    x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs_expanded)
    skips = [x]
    n = layers.UpSampling1D(size=num_dim)(time_embedding)
    x = layers.Concatenate()([x, n])

    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width,stride,kernel, attention,use_skip=True)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1],kernel, attentions[-1])(x)

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width,stride, attention,use_skip=True)([x, skips])

    x = layers.Concatenate()([x, skips.pop()])
    x = layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)

    outputs = tf.reshape(x,(-1,num_dim))
    # if use_clamp:
    #     outputs = soft_clamp(outputs)

    return inputs, outputs




def get_encoder(
        num_dim,
        end_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        use_clamp=False,
):
    
    inputs = keras.Input((num_dim))
    inputs_expanded = layers.Reshape((num_dim,1))(inputs)    
    x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs_expanded)
    n = layers.UpSampling1D(size=num_dim)(time_embedding)
    x = layers.Concatenate()([x, n])

    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width,stride,kernel, attention)(x)
    x = ResidualBlock(widths[-1],kernel, attentions[-1])(x)
    representation = layers.Flatten()(x)
            
    z_mean = tfa.layers.SpectralNormalization(
        layers.Dense(end_dim, name="z_mean",kernel_initializer="zeros"))(representation)
    z_mean = soft_clamp(z_mean)            
    z_log_sig = tfa.layers.SpectralNormalization(
        layers.Dense(end_dim, name="z_log_var",kernel_initializer="zeros"))(representation)
    z_log_sig = soft_clamp(z_log_sig)
                
    return inputs,z_mean,z_log_sig

def get_decoder(
        num_dim,
        end_dim,
        cond_embedding,
        mlp_embed=64,
        num_layer=5,
        mlp_dim=1024,
        zeros=False,
        activation='swish'
):

    
    if activation == 'swish':
        act=keras.activations.swish
    elif activation == 'leakyrelu':
        act = layers.LeakyReLU(alpha=0.01)


    def resnet_dense(input_layer,hidden_size,nlayers=2):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            layer=act(layers.Dense(hidden_size,activation=None)(layer))
        return (layer + residual)/np.sqrt(2)

    inputs = keras.Input((num_dim))    
    embed = act(layers.Dense(mlp_embed)(cond_embedding))
    residual = act(layers.Dense(mlp_dim)(inputs))
    
    for _ in range(num_layer-1):
        residual =  resnet_dense(tf.concat([residual,embed],-1),mlp_dim)

    outputs = act(layers.Dense(2*end_dim)(residual))
    outputs = layers.Dense(end_dim)(outputs)
    outputs = soft_clamp(outputs)
    return inputs,outputs




def get_MLP(
        num_dim,
        end_dim,
        cond_embedding,
        mlp_embed=64,
        num_layer=5,
        mlp_dim=1024,
        zeros=True,
        use_clamp=True,
        activation='leakyrelu'
):

    
    if activation == 'swish':
        act=keras.activations.swish
    elif activation == 'leakyrelu':
        act = layers.LeakyReLU(alpha=0.01)


    def resnet_dense(input_layer,hidden_size,nlayers=2):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            layer=act(layers.Dense(hidden_size,activation=None)(layer))
        return layer + residual
    

    inputs = keras.Input((num_dim))
    residual = act(layers.Dense(2*mlp_dim)(tf.concat([inputs,cond_embedding],-1)))    
    residual = layers.Dense(mlp_dim)(residual)

    layer = residual
    for _ in range(num_layer-1):
        embed = layers.Dense(mlp_embed)(cond_embedding)
        layer =  resnet_dense(tf.concat([layer,embed],-1),mlp_dim)

    layer = layers.Dense(mlp_embed)(layer+residual)
    
    if zeros:
        outputs = layers.Dense(end_dim,kernel_initializer="zeros")(residual)
    else:
        outputs = layers.Dense(end_dim)(residual)
        
    if use_clamp:
        outputs = soft_clamp(outputs,1.0)

    return inputs,outputs

