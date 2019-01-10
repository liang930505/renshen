import cv2
import os
from skimage import color, morphology, feature
import numpy as np

path = '/home/gszn/rename/'

p = os.listdir(path)
# 图像路径
img_path = path + '58.jpg'
# 读取彩色图像
img1 = cv2.imread(img_path)

# 读取灰度图像
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 四种二值化方法， 选择第四种OTSU二值化
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 开运算
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(th4, cv2.MORPH_OPEN,  kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# 提取轮廓
image, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 迭代所有轮廓数组， 寻找直边外接矩形最大轮廓
areas = []
for cnts in contours:
    x, y, w, h = cv2.boundingRect(cnts)
    area = w * h
    areas.append(area)
# 获得最大轮廓索引
index = areas.index(max(areas))
print(index)
# 获得最大轮廓
cnt = contours[index]

# 获得矩形位置点以及长和宽
x, y, w, h = cv2.boundingRect(cnt)
# 数组切片，剪裁图像
crop_img = img1[y: y + h, x: x + w]
# 保存图像
cv2.imwrite('crop_58.jpg', crop_img)
# 绘制直边外接矩形
cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 显示绘制图像
cv2.imshow('show', img1)
cv2.waitKey()
