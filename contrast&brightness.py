'''
Description   : 调节图片亮度和对比度（实验性）
Time          : 2024/10/30 11:29:20
Author        : 宽后藤 
'''

import cv2
import numpy as np
import os

def over_exposure(img):
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=100)

path = 'C:/Users/29560/Desktop/babaji/'
save_path = 'C:/Users/29560/Desktop/lucitic/'

file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name))

    img = over_exposure(img)

    # plt.figure()
    # plt.subplot(1,3,1), plt.imshow(img), plt.axis('off'), plt.title('origin')
    # plt.subplot(1,3,2), plt.imshow(img_rgb_enhanced), plt.axis('off'), plt.title('equalized')
    # plt.subplot(1,3,3), plt.imshow(img_gammaed), plt.axis('off'), plt.title('enhanced')
    # plt.show()
    cv2.imwrite(os.path.join(save_path, imgs.name), img)
    # cv2.imwrite(os.path.join('C:/Users/29560/Desktop/', imgs.name), img_gammaed)

print('done')