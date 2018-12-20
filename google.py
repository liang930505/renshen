from keras.layers import Dense, Flatten, Dropout, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import cv2
import keras
import os
from keras.layers import BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import TensorBoard, LearningRateScheduler,ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import datetime
from keras.models import load_model
from skimage import io
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import Sequential
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))


dropout = 0.5
weight_decay = 0.0001

path = '/home/gszn/resize/'
pics = os.listdir(path)

image_width = 80
image_height = 60


def img_read(sub_path):
    str1 = path + '/*.jpg'
    dataX = io.ImageCollection(str1)
    train_xdata = np.array([dataX])
    train_xdata = train_xdata[0]

    return train_xdata


def data_preprocess(data):
    x_scaled = data / 255
    return x_scaled


train_data = img_read(path)

label = pd.read_excel('/home/gszn/label.xlsx', header=None)

np_class = 3

# permutation = np.random.permutation(train_data.shape[0])
# print(permutation)
# shuffle_data = train_data[permutation, :, :, :]
#
# label = label.values
# shuffle_label = label[permutation, :]
#
#
# x_train = shuffle_data[0:math.floor(len(shuffle_data) * 0.8), :, :, :]
# x_test = shuffle_data[math.floor(len(shuffle_data) * 0.8):, :, :, :]
# y_train = shuffle_label[0:math.floor(len(shuffle_label) * 0.8), :]
# y_test = shuffle_label[math.floor(len(shuffle_label) * 0.8):, :]
# y_test = pd.DataFrame(y_test)

x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2)
print(y_train)
input_shape = x_train[0].shape
print(y_train[100:101])
io.imsave('/home/gszn/ren.jpg', x_test[100])

x_train = data_preprocess(x_train)
x_test = data_preprocess(x_test)


model = Sequential()
model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(np_class, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001, epsilon=1e-06)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
log_dir = '/home/gszn/PycharmProjects/logs'
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
model.fit(x_train, y_train, batch_size=100, epochs=500, validation_data=(x_test, y_test), verbose=1, callbacks=[logging, checkpoint])

model.save('vgg19_model10.h5')

def test_accuracy():
    model = load_model('vgg19_model10.h5')
    pred = model.predict(x_test)
    print(pred)
    # y_pred = []
    # y_true = []
    # for each_list in pred:
    #     pred_label = each_list.argsort()
    #     y_pred.append(pred_label[8])
    # for l in range(len(y_test)):
    #     true_label = y_test.iloc[l].argsort()
    #     y_true.append(true_label[8])
    #
    # true_account = []
    # for i in range(len(y_pred)):
    #     if int(y_pred[i]) == int(y_true[i]):
    #         true_account.append(1)
    # accuracy = len(true_account)/len(y_pred)
    # accuracy1 = len(true_account)/len(y_true)
    # print(accuracy)
    # print(accuracy1)


test_accuracy()

