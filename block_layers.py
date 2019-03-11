from keras.layers import Conv2D, MaxPooling2D, Input, ReLU,UpSampling2D,Dropout, concatenate,BatchNormalization,ReLU,Activation
from keras.optimizers import Adam
from keras.models import Model
import numpy as np



def down_conv_block(conv_iuput,filters,dropout=False,pool=True):
    X = Conv2D(filters=filters, kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal')(conv_iuput)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    conv_block = Activation('relu')(X)
    if dropout:
        conv_block = Dropout(0.5)(X)
    if pool:
        X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_block)

    return X, conv_block


def up_conv_block(input, conn_block,filters):
    up = UpSampling2D(size=(2,2))(input)
    conv_block = Conv2D(filters=filters, kernel_size=(2,2),strides=(1,1), padding='same', kernel_initializer='he_normal')(up)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Activation('relu')(conv_block)
    conv = concatenate([conn_block,conv_block],axis=3)
    conv = Conv2D(filters=filters,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters=filters,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer='he_normal')(conv)

    return conv