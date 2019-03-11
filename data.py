import cv2
import os
import sys
import h5py
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


BASE_PATH = "data/membrane/"


# 将图片储存入hdf5，这里因为图片数据量很小，直接每次循环遍历就好
def save_data_to_h5(X_data, Y_data,filename,X_name,Y_name):
    if not os.path.exists('h5_file'):
        os.mkdir('h5_file')
    if not os.path.exists(os.path.join('h5_file',filename)):
        # file = h5py.File(os.path.join(BASE_PATH,'h5_file',filename))
        with h5py.File(os.path.join('h5_file',filename)) as file:
            file.create_dataset(name=X_name,data=X_data)
            file.create_dataset(name=Y_name,data=Y_data)


# 从h5文件中导入数据
def load_data_from_h5(filename):
    filepath = os.path.join('h5_file',filename)
    if not os.path.exists(filepath):
        print('%s is not exist..' % filename)
        sys.exit()
    
    file = h5py.File(filepath, 'r')
    X_train = file['images']
    Y_train = file['masks']
    # file.close()
    # print(X_train.shape, Y_train.shape)
    return X_train, Y_train


# 由于我已导出图片，不必要每次遍历图片，故注释以下
# 转换图像数据（已储存HDF5可省略此步）
# def cover_image_to_data(path):
#     # 第二个参数为0意思是以单通道模式读取灰度图片
#     img = cv2.imread(path,0)
#     img_resize = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
#     img_data = np.array(img_resize, dtype='float32')
#     img_data = np.reshape(img_data, img_data.shape+(1,))
#     img_data = np.expand_dims(img_data, 0)
#     return img_data


# 整理训练数据（已储存HDF5可省略此步）
# def load_train_data():
#     img_data_list = []
#     lb_data_list = []
#     for i in range(30):
#         # 读取训练图片
#         img_data = cover_image_to_data(os.path.join(BASE_PATH,'train','image',str(i)+'.png'))
#         # print(img_data[0,0,0,0])
#         # 读取训练标记图片
#         lb_data = cover_image_to_data(os.path.join(BASE_PATH,'train','label',str(i)+'.png'))
#         img_data_list.append(img_data)
#         lb_data_list.append(lb_data)

#     X = np.concatenate(img_data_list)
#     Y = np.concatenate(lb_data_list)
#     save_data_to_h5(X,Y,'train.hdf5','images','masks')
#     return X,Y


# 整理验证数据(已储存HDF5可省略此步)
# def load_test_data():
#     img_data_list = []
#     lb_data_list = []

#     for i in range(30):
#         # 读取验证图片
#         img_data = cover_image_to_data(os.path.join(BASE_PATH,'test',str(i)+'.png'))
#         # 读取验证标记图片
#         lb_data = cover_image_to_data(os.path.join(BASE_PATH,'test',str(i)+'_predict.png'))
#         img_data_list.append(img_data)
#         lb_data_list.append(lb_data)

#     X = np.concatenate(img_data_list)
#     Y = np.concatenate(lb_data_list)
#     save_data_to_h5(X,Y,'test.hdf5','images','masks')
#     return predata(X, Y)


# 数据流
def data_flow(X_train, Y_train):

    # 数据增强选项
    datagen = ImageDataGenerator(
        rotation_range=0.2,      # 旋转图像0-180度
        width_shift_range=0.05,  # 水平平移图像（基于图像宽度比例）
        height_shift_range=0.05, # 垂直平移图像（基于图像高度比例）
        horizontal_flip=True,    # 水平翻转图像
        shear_range=0.05,        # 水平或垂直投影变换
        zoom_range=0.05,         # 缩放
        fill_mode='nearest'      # 超出边界
        )

    # X_train, Y_train = load_data_from_h5('train.hdf5')
    image_generator = datagen.flow(X_train, batch_size=2,seed=1)
    mask_generator = datagen.flow(Y_train, batch_size=2,seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img, mask = predata(img,mask)
        yield (img, mask)
    # data_iter = datagen.flow(X_train, Y_train, batch_size=2)
    # return data_iter

# 数据预处理
def predata(X,Y):
    X /= 255.
    Y /= 255.
    Y[Y>0.5] = 1
    Y[Y<=0.5] = 0
    return X, Y


# 图像预处理
def preprocess_image(img_path, model_image_size):
    image = cv2.imread(img_path,0)
    size = image.shape
    img_shape = (float(size[0]),float(size[1]))
    image_resize = cv2.resize(image,model_image_size,interpolation=cv2.INTER_CUBIC)
    image_data = np.reshape(image_resize, image_resize.shape+(1,))
    image_data = np.array(image_data, dtype='float32')
    image_data = np.expand_dims(image_data, 0)
    image_data /= 255.
    return image, image_data, img_shape


# 矩阵转图像
def matrix_to_Image(data):
    data = data[0]
    one_channel_data = data*255
    img_data = np.concatenate(( one_channel_data,
                                one_channel_data,
                                one_channel_data),
                                axis=-1)
    new_im = Image.fromarray(img_data.astype(np.uint8))
    return new_im

