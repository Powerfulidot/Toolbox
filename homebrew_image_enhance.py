'''
Description   : ACLAHE-AGAMMA-小波降噪
Time          : 2024/12/11 16:30:54
Author        : 宽后藤
'''

import cv2
import numpy as np
import math
import os
import pywt

def CLAHE(img):                                                  # 自适应直方图均衡
    # img = img.astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h = img_hsv[:,:,0]
    img_s = img_hsv[:,:,1]
    img_v = img_hsv[:,:,2]
    mean, std_dev = cv2.meanStdDev(img_gray)
    clip_limit = float(std_dev ** 2 / (mean * 2))
    # print(clip_limit)                                           # cliplimit现在可自适应获得
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

def WAVELET_DENOISE(img):                                       # 将图像小波分解，对高频应用高斯滤波以降噪
    reform_channels = []
    channels = cv2.split(img)
    # img_b = img[:,:,0]
    # img_g = img[:,:,1]
    # img_r = img[:,:,2]
    for channel in channels:
        cA, (cH, cV, cD) = pywt.dwt2(channel, 'haar')

        cH = cv2.GaussianBlur(cH, (3, 3), 0)
        cV = cv2.GaussianBlur(cV, (3, 3), 0)
        cD = cv2.GaussianBlur(cD, (3, 3), 0)
        # cH = cv2.bilateralFilter(cH, 5, 150, 150)
        # cV = cv2.bilateralFilter(cV, 5, 150, 150)
        # cD = cv2.bilateralFilter(cD, 5, 150, 150)

        channel = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        reform_channels.append(channel/np.max(channel) * 255)

    img_reformed = cv2.merge(reform_channels).astype(np.uint8)
    return img_reformed


path = r'C:/Users/29560/Desktop/lucitic/'
save_path = r'C:/Users/29560/Desktop/oi/'

file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name))

    img_gammaed_1 = AGAMMA(img)
    img_clahed = CLAHE(img)
    img_denoised = WAVELET_DENOISE(img_clahed)
    img_gammaed_2 = AGAMMA(img_denoised)
    img = cv2.addWeighted(img_gammaed_1, 0.5, img_gammaed_2, 0.5, 0)
    img = AGAMMA(img)

    cv2.imwrite(os.path.join(save_path, imgs.name), img)
    print(imgs.name)

print('done')