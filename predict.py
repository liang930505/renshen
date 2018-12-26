from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import Sequential
import datetime
import threading
import shutil

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# 读取文件夹内后缀为‘JPG’、‘png’、‘jpeg’、‘bmp’图像，将图像大小缩放至80 × 60，如果存在图像未被正确读取的问题，则打印错误信息，程序继续运行
# 函数返回data和train_data,data为待预测图像数据，train_data为与data对应的文件名

def img_read(sub_path):

    train_data = []
    file = os.listdir(sub_path)
    for name in file:
        name_list = name.split('.')
        if name_list[-1] == 'jpg':
            train_data.append(name)
        if name_list[-1] == 'png':
            train_data.append(name)
        if name_list[-1] == 'jpeg':
            train_data.append(name)
        if name_list[-1] == 'bmp':
            train_data.append(name)
    img_list = []

    data = np.empty((len(train_data), 60, 80, 3), dtype='float32')
    for i in range(len(train_data)):
        pic = cv2.imread(sub_path + train_data[i])
        # 用try...except处理cv2.imread()读取图片为空的问题
        try:
            scaled_pic = cv2.resize(pic, (80, 60), interpolation=cv2.INTER_CUBIC)
            pics = np.asarray(scaled_pic, dtype='float32')
            img_list.append(train_data[i])
            data[i, :, :, :] = pics
        except cv2.error:
            print(i, 'time error')
            pass
    return data, train_data


# 建立模型，加载训练好的模型权重，函数参数为待预测数据集，返回值为模型对象

def build_model(x_test):
    input_shape = x_test[0].shape
    # 网络结构：四层卷基层、三层池化层、两层全连接层
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.load_weights('/home/gszn/PycharmProjects/log/logsep175-loss0.027-val_loss0.069.h5')
    return model

# 预测功能，参数x_test 为待预测图像集，model为模型对象，取预测结果最大值索引作为类别标签，将图像文件名和图像类别以key-value字典的形式\
# 逐一添加到列表中，返回值为一个含有多个字典的列表。

def predict(x_test, model):
    y_pred = []
    pred = model.predict(x_test)
    for j in range(len(pred)):
        # 数组转列表
        pred_list = pred[j].tolist()
        # 列表最大值索引
        c = pred_list.index(max(pred[j]))
        if c == 0:
            y_pred.append({name_list[j]: 'big'})
            # 移动图像到所属类别文件夹
            shutil.move(path+name_list[j], '/home/gszn/big/')
        if c == 1:
            y_pred.append({name_list[j]: 'middle'})
            shutil.move(path + name_list[j], '/home/gszn/middle/')
        if c == 2:
            y_pred.append({name_list[j]: 'small'})
            shutil.move(path + name_list[j], '/home/gszn/small/')
    print(y_pred)


# 程序入口

if __name__ == '__main__':

    start = datetime.datetime.now()
    # 图片所在路径
    path = '/home/gszn/'
    # path1 = '/home/gszn/1/'
    # path2 = '/home/gszn/2/'
    # path3 = '/home/gszn/3/'
    # t1 = threading.Thread(target=img_read, args=path1)
    # threads.append(t1)
    # t2 = threading.Thread(target=img_read, args=path2)
    # threads.append(t2)
    # t3 = threading.Thread(target=img_read, args=path3)
    # threads.append(t3)
    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()

    # 载入图像
    img, name_list = img_read(path)
    pic_end = datetime.datetime.now()
    print('pictures read end, need time:', pic_end - start, 's')

    img = img / 255
    print('predict picture numbers is :', img.shape[0])
    model_end = datetime.datetime.now()
    # 加载模型
    model = build_model(img)
    print('build model end, need time:', model_end - pic_end, 's')
    # 开始预测
    predict(img, model)
    end = datetime.datetime.now()
    print('Predict Need Time:', end - model_end, 's')
