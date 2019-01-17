import os
import random
import cv2
from multiprocessing import Process, Queue, Pipe
import classification_model
import datetime
import time


# cv2读取图像,将图像数据、摄像头及图像名称、路径以元组的形式加入队列
def get_image(q, num, name='camera'):
    '''
    :param q: 队列名称
    :param path: 图像所在路径
    :param name: 摄像头名称，若不填写则默认为’camere‘
    :return: 无返回值
    '''
    num = int(num)
    capture = cv2.VideoCapture(num)
    if capture.isOpened():
        while True:
            capture.set(3, 1024)
            capture.set(4, 1280)
            ret, image = capture.read()

            s = datetime.datetime.now()
            # img = cv2.imread(path + file_list[0])
            data = (image, name)
            q.put(data)
            if q.full() == True:
                time.sleep(1)
            else:
                # file_list.remove(file_list[0])
                e = datetime.datetime.now()
                print('获取图像所需时间：', e - s, q.qsize())
    else:
        print('摄像头%s未开启' % num)


# 在硬盘上存储图像，在存储路径下自动创建大、中、小三个文件夹，根据图像对应类别，分别存储
def save_file(con2, ppath):
    '''

    :param con2: 管道名称
    :param ppath: 存储路径
    :return: 无返回值
    '''
    # 输入文件存储路径

    if not os.path.exists(ppath):
        os.mkdir(ppath)
    # 在存储路径下创建三个分类文件夹
    big_file = ppath + '/' + 'big_file/'
    middle_file = ppath + '/' + 'middle_file/'
    small_file = ppath + '/' + 'small_file/'
    if not os.path.exists(big_file):
        os.mkdir(big_file)
    if not os.path.exists(middle_file):
        os.mkdir(middle_file)
    if not os.path.exists(small_file):
        os.mkdir(small_file)
    # 根据队列q1包含的类别，将图片存储在大中小三个文件夹内
    i = 0
    while True:
        if con2.recv != None:
            start = datetime.datetime.now()
            image = con2.recv()
            print(os.getpid(), type(image[0]), type(image))
            if image[1] == 0:
                cv2.imwrite(big_file + image[0] + '_' + str(i) + '.jpg', image[2])
            elif image[1] == 1:
                cv2.imwrite(middle_file + image[0] + '_' + str(i) + '.jpg', image[2])
            else:
                cv2.imwrite(small_file + image[0] + '_' + str(i) + '.jpg', image[2])
            # return file_list
            end = datetime.datetime.now()
            print('一张图片保存需要时间:', end - start)
            i = i + 1
        else:
            time.sleep(1)


# 创建六个进程同步执行get_image函数，创建一个进程执行save_file函数

def que(model_inf):
    (con1, con2) = Pipe()
    q = Queue(1000)

    p1 = Process(target=get_image, args=(q, 0, 'camera1'))
    p1.start()
    p2 = Process(target=get_image, args=(q, 1, 'camera2'))
    p2.start()
    # p3 = Process(target=get_image, args=(q, path_list[2], 'camera3'))
    # p3.start()
    # p4 = Process(target=get_image, args=(q, path_list[3], 'camera4'))
    # p4.start()

    # p5 = Process(target=get_image, args=(q, path_list[4], 'camera5'))
    # p5.start()
    #
    # p6 = Process(target=get_image, args=(q, path_list[5], 'camera6'))
    # p6.start()
    p7 = Process(target=save_file, args=(con2, '/home/gszn/save/'))
    p7.start()

    while True:
        # print('ori:', q.qsize())
        if q.empty() != True:
            data = q.get(timeout=1)
            # print(q.qsize())
            try:
                start = datetime.datetime.now()
                label = classification_model.predict(data[0], model_inf)
                con1.send((data[1], label, data[0]))
                end = datetime.datetime.now()
                print('发送管道:', end - start)
            except:
                pass

        else:
            # print('队列为空')
            continue


if __name__ == '__main__':
    model_inf = classification_model.open_model()

    que(model_inf)