'''
@Description   : 使用二值化和开闭操作尝试进行边缘检测，鲁棒性差
@Time          : 2024/10/30 11:33:49
@Author        : 宽后藤 
'''

import cv2
import numpy as np


path = '2.png'
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

kernel_small = np.ones((1, 5))
kernel_big = np.ones((4, 4))
# edges = cv2.GaussianBlur(edges, (5, 5), 0) # 高斯平滑
# img_di = cv2.dilate(edges, kernel_small, iterations=5) # 膨胀5次
img_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_big) # 闭操作
img_close = cv2.GaussianBlur(img_close, (5, 5), 0) # 高斯平滑
_, img_bin = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY) # 二值化

cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.namedWindow('4090', cv2.WINDOW_NORMAL)
cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)

cv2.imshow('origin', img)
cv2.imshow('4090', edges)
cv2.imshow('babaji', img_bin)
# cv2.imshow("lucitic", rotate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()