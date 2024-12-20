'''
Description   : 自适应直方图均衡和自适应伽马变换的脚本
Time          : 2024/10/30 11:28:49
Author        : 宽后藤 
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def CLAHE(img):                                                  # 自适应直方图均衡
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h = img_hsv[:,:,0]
    img_s = img_hsv[:,:,1]
    img_v = img_hsv[:,:,2]
    mean, std_dev = cv2.meanStdDev(img_gray)
    clip_limit = float(std_dev ** 2 / (mean * 2))
    print(clip_limit)
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = (15, 15))  
    img_v = clahe.apply(img_v)
    img_hsv_enhanced = cv2.merge([img_h, img_s, img_v])
    img_rgb_enhanced = cv2.cvtColor(img_hsv_enhanced, cv2.COLOR_HSV2BGR)
    return img_rgb_enhanced

def AGAMMA(img):                                                 # 自适应伽马矫正
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    mean_b = np.mean(img_b)
    mean_g = np.mean(img_g)
    mean_r = np.mean(img_r)
    gamma_b = math.log10(0.5) / math.log10(mean_b / 255)
    gamma_g = math.log10(0.5) / math.log10(mean_g / 255)
    gamma_r = math.log10(0.5) / math.log10(mean_r / 255)
    gamma_rectify = (gamma_b + gamma_g + gamma_r) / 3

    gamma_table = [np.power(x / 255.0, gamma_rectify) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    img_gammaed = cv2.LUT(img, gamma_table)
    return img_gammaed


path = 'C:/Users/29560/Desktop/lucitic/'
save_path = 'C:/Users/29560/Desktop/huh/'

file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name))

    # img = Image.open('C:/Users/29560/Desktop/test/dark_3.jpg')
    # img = np.uint8(img)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # 分离HSV通道
    # img_h = img_hsv[:,:,0]
    # img_s = img_hsv[:,:,1]
    # img_v = img_hsv[:,:,2]

    # img_v = cv2.equalizeHist(img_v)                                 # 直方图均衡
    # img_hsv_equalized = cv2.merge([img_h, img_s, img_v])
    # img_rgb_equalized = cv2.cvtColor(img_hsv_equalized, cv2.COLOR_HSV2BGR)

    img = CLAHE(img)
    img = AGAMMA(img)

    # plt.figure()
    # plt.subplot(1,3,1), plt.imshow(img), plt.axis('off'), plt.title('origin')
    # plt.subplot(1,3,2), plt.imshow(img_rgb_enhanced), plt.axis('off'), plt.title('equalized')
    # plt.subplot(1,3,3), plt.imshow(img_gammaed), plt.axis('off'), plt.title('enhanced')
    # plt.show()
    cv2.imwrite(os.path.join(save_path, imgs.name), img)
    print(imgs.name)
    # cv2.imwrite(os.path.join('C:/Users/29560/Desktop/', imgs.name), img_gammaed)

print('done')