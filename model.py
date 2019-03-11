from keras.layers import Conv2D, MaxPooling2D, Input, ReLU,UpSampling2D,Dropout, concatenate,BatchNormalization,ReLU,Activation
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
from block_layers import down_conv_block, up_conv_block

def unetModel(input_shape):

    X_input = Input(input_shape)

    # 块1
    X, conv_block1 = down_conv_block(X_input, 64)

    # 块2
    X, conv_block2 = down_conv_block(X, 128)

    # 块3
    X, conv_block3 = down_conv_block(X, 256)

    # 块4
    X, conv_block4 = down_conv_block(X, 512, dropout=True)

    # 块5
    X, conv_block5 = down_conv_block(X, 1024, dropout=True)

    # 块6: 块5做上采样 链接 块4做卷积
    conv6 = up_conv_block(conv_block5,conv_block4,filters=512)

    # 块7： 块6做上采样 链接 块3做卷积
    conv7 = up_conv_block(conv6,conv_block3,filters=256)

    # 块8： 块7做上采样 链接 块2做卷积
    conv8 = up_conv_block(conv7,conv_block2,filters=128)

    # 块9： 块8做上采样 链接 块1做卷积
    conv9 = up_conv_block(conv8,conv_block1,filters=64)
    
    conv9 = Conv2D(filters=2,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer='he_normal', activation='relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    conv10 = Conv2D(filters=1,kernel_size=(1,1), strides=(1,1),padding='same',kernel_initializer='he_normal', activation='sigmoid')(conv9)
    

    model = Model(inputs = X_input, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model