from data import matrix_to_Image, preprocess_image
from keras.models import load_model
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #不加这个，macOS系统会有好几个警告，但是不影响程序运行

if __name__ == '__main__':
    if len(sys.argv) == 2:
        model = load_model('unet_membrane.hdf5')
        image, image_data, img_shape = preprocess_image('data/membrane/test/7.png',(256,256))
        img_data = model.predict(image_data)
        img = matrix_to_Image(img_data)
        img.show()
    else:
        print('Fail: Please fill in the image path.')