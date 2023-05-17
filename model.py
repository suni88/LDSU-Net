# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:44:42 2020

@author: mail2
"""

from __future__ import print_function
import numpy as np 
import os
import skimage.io as io
#import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
#import skimage.transform as trans
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Lambda,SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, Normalization
from keras.optimizers import RMSprop
from keras.regularizers import l2
#from sklearn.model_selection import train_test_split
#def get_unet(self):

		#inputs = Input((self.img_rows, self.img_cols, 1))
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = SeparableConv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)
    

    return x

def Mainblock(U, inp, alpha = 1.67):
    '''
    Main Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')
    conv3x3 = conv2d_bn(conv3x3, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(inp, int(W*0.333), 3, 3,
                        activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv5x5, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    #conv7x7= MaxPooling2D((1,1), strides=(1,1))(inp)
    conv7x7 = conv2d_bn(inp, int(W*0.333), 3, 3,
                        activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv7x7, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    return out
def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = SeparableConv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = SeparableConv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = SeparableConv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = SeparableConv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = SeparableConv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn
		
def udnet(pretrained_weights = None,input_size = (256,256,1)):
   inputs = Input(input_size)		
   conv1_1 =SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
   BatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
   ReLU1_1 = Activation('relu')(BatchNorm1_1)
   conv1_2 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU1_1)
   drop1_2 = Dropout(0)(conv1_2)
   Merge1 = concatenate([conv1_1,drop1_2], axis = 3)
   Merge1= Mainblock (32, Merge1)
   pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1)
   conv2_1 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
   BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
   ReLU2_1 = Activation('relu')(BatchNorm2_1)
   conv2_2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
   drop2_2 = Dropout(0)(conv2_2)
   Merge2 = concatenate([conv2_1,drop2_2], axis = 3)
   Merge2= Mainblock (64, Merge2)
   pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)
   conv3_1 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
   BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
   ReLU3_1 = Activation('relu')(BatchNorm3_1)
   conv3_2 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
   drop3_2 = Dropout(0)(conv3_2)
   Merge3 = concatenate([conv3_1,drop3_2], axis = 3)
   Merge3= Mainblock (128, Merge3)
   pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)
   conv4_1 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
   BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
   ReLU4_1 = Activation('relu')(BatchNorm4_1)
   conv4_2 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
   drop4_2 = Dropout(0)(conv4_2)
   Merge4 = concatenate([conv4_1,drop4_2], axis = 3)
   drop4 = Dropout(0.5)(Merge4)
   drop4= Mainblock (256, drop4)
   pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
   conv5_1 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
   BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
   ReLU5_1 = Activation('relu')(BatchNorm5_1)
   conv5_2 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
   drop5_2 = Dropout(0)(conv5_2)
   Merge5 = concatenate([conv5_1,drop5_2], axis = 3)
   drop5 = Dropout(0.5)(Merge5)
#merge8 = concatenate([conv2,up8], axis = 3)
   
   gating_16 = gating_signal(drop5, 512, batch_norm= False)
   att_16 = attention_block(drop4, gating_16, 512)
   up6 = SeparableConv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
   merge6 = concatenate([att_16,up6], axis = 3)
   conv6_1 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
   BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
   ReLU6_1 = Activation('relu')(BatchNorm6_1)
   conv6_2 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
   drop6_2 = Dropout(0)(conv6_2)
   Merge6 = concatenate([conv6_1,drop6_2],axis = 3)
   
   gating_32 = gating_signal(Merge6, 256, batch_norm= False)
   att_32 = attention_block(Merge3, gating_32, 256)
   up7 = SeparableConv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
   merge7 = concatenate([att_32,up7],axis = 3)
   conv7_1 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
   BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
   ReLU7_1 = Activation('relu')(BatchNorm7_1)
   conv7_2 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
   drop7_2 = Dropout(0)(conv7_2)
   Merge7 = concatenate([conv7_1,drop7_2], axis = 3)
   
   gating_64 = gating_signal(Merge7, 128, batch_norm= False)
   att_64 = attention_block(Merge2, gating_64, 128)
   up8 = SeparableConv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
   merge8 = concatenate([att_64,up8], axis = 3)
   conv8_1 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
   BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
   ReLU8_1 = Activation('relu')(BatchNorm8_1)
   conv8_2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
   drop8_2 = Dropout(0)(conv8_2)
   Merge8 = concatenate([conv8_1,drop8_2], axis = 3)
   
   gating_128 = gating_signal(Merge8, 64, batch_norm= False)
   att_32 = attention_block(Merge1, gating_128, 64)
   up9 = SeparableConv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
   merge9 = concatenate([att_32,up9], axis = 3)
   conv9_1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
   BatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv9_1)
   ReLU9_1 = Activation('relu')(BatchNorm9_1)
   conv9_2 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU9_1)
   drop9_2 = Dropout(0)(conv9_2)
   Merge9 = concatenate([conv9_1,drop9_2],axis = 3)

   conv9 = SeparableConv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
   conv10 = SeparableConv2D(1, 1, activation = 'sigmoid')(conv9)#sigmoid
                #conv10 = Conv2D(1, 1, activation = 'softmax')(conv9)#sigmoid

   model = Model(inputs = inputs, outputs = conv10)

   model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
                #model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
   model.summary()
   


   if(pretrained_weights):
    	model.load_weights(pretrained_weights)

   return model
