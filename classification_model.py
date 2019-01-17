import numpy as np
import os
import cv2
import sys
import tensorflow as tf 
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib as _device_lib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def vgg16(inputs):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],activation_fn=tf.nn.relu):

		net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], stride=1, scope='conv1')
		net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool1')
		net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3],  stride=1, scope='conv2')
		net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool2')
		net = slim.repeat(net, 1, slim.conv2d, 256, [3, 3],  stride = 1, scope='conv3')
		net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool3')
		net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],  stride = 1, scope='conv4')
		net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool4')
		net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],  stride = 1, scope='conv5')
		net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool5')

		net = tf.reshape(net, (-1, 6*8*512))

		net = slim.fully_connected(net, 2048, scope='fc6')
		net = slim.fully_connected(net, 256, scope='fc7')
		net = slim.fully_connected(net, 3, activation_fn=None, scope='fc8')

	return net


def open_model():

	#定义占位符
	x = tf.placeholder (tf.float32, [None, 192, 256, 3])

	#定义模型
	y_ = vgg16(x)
	print('模型已加载!')

	#分类
	y_number = tf.argmax(y_, 1)


	#设置图的信息为使用最小显存
	# tf_config = tf.ConfigProto()
	# tf_config.gpu_options.allow_growth = True


	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	saver = tf.train.Saver()
	kpt = tf.train.latest_checkpoint('/home/gszn/PycharmProjects/renshen/Panaxginseng/train_model_data/')


	# sess = tf.Session(config=sess)
	print('计算图打开!')
	print(_device_lib.list_local_devices()[1].physical_device_desc)

	sess.run(tf.global_variables_initializer())


	if kpt != None:
		saver.restore(sess, kpt)
		print('参数载入成功!')
	else:
		print('参数载入失败')
		sys.exit()


	return (sess, y_number, x)


def predict(img, model_inf):

	img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis=0)

	return model_inf[0].run([model_inf[1]], feed_dict={model_inf[2]: img})[0][0]


if __name__ == '__main__':
	pass
