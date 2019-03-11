import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from PIL import Image
import sys, getopt

from data import data_flow, load_data_from_h5, predata, preprocess_image
from model import unetModel

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #不加这个，macOS系统会有好几个警告，但是不影响程序运行
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


if __name__ == '__main__':

    opts, args = getopt.getopt(sys.argv[1:], "f:")
    is_first = ''
    for op, value in opts:
        if op == "-f":
            is_first = value


    model = unetModel((256,256,1))

    X_train, Y_train = load_data_from_h5('train.hdf5')
    images, masks = load_data_from_h5('test.hdf5')

    data_iter = data_flow(X_train, Y_train)
    images = np.array(images)
    masks = np.array(masks)
    X_test, Y_test = predata(images, masks)
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
    if is_first == 'yes':
        
        model.fit_generator(data_iter,
                            steps_per_epoch=300,
                            epochs=10,
                            validation_data=(X_test, Y_test),
                            # workers=4,
                            callbacks=[model_checkpoint]
                            )
    elif is_first=='no':
        model = load_model('unet_membrane.hdf5')
        loss,accuracy = model.evaluate(X_test,Y_test)
        print('\n对上一次模型性能评估：')
        print('\ttest loss',loss)
        print('\taccuracy',accuracy)
        print()
            
        model.fit_generator(data_iter,
                            steps_per_epoch=600,
                            epochs=10,
                            validation_data=(X_test, Y_test),
                            callbacks=[model_checkpoint]
                            )
