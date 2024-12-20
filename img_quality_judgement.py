'''
Description   : 图像质量的评价指标（标准差、平均梯度和信息熵）
Time          : 2024/11/20 11:01:42
Author        : 宽后藤 
'''

import cv2
import numpy as np
import math
import os

def standard_deviation(img):                # 标准差（标准差越大，灰度分布越分散）
    _, std_dev = cv2.meanStdDev(img)

    return std_dev

def mean_gradient(img):                     # 平均梯度（平均梯度越大，边缘越明显，间接反映清晰度）
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    m_g = np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2))

    return m_g

def entrophy(img):                          # 信息熵（信息熵越大，信息量越丰富）
    tmp = [0] * 256
    val = 0
    k = 0
    res = 0

    img = np.array(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i]/ k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    return res

path = 'C:/Users/29560/Desktop/huh/'
std_dev_sum = 0
m_g_sum = 0
res_sum = 0

file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name), 0)

    std_dev = standard_deviation(img)
    m_g = mean_gradient(img)
    res = entrophy(img)    

    std_dev_sum += std_dev
    m_g_sum += m_g
    res_sum += res

std_dev = std_dev_sum / 6
m_g = m_g_sum / 6
res = res_sum / 6

print('标准差：', std_dev, '\n平均梯度：', m_g, '\n信息熵：', res)