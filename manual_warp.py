'''
Description   : 手动选点的仿射变换
Time          : 2024/10/30 11:31:26
Author        : 宽后藤 
'''

import cv2
import numpy as np

img = cv2.imread('C:/Users/29560/Desktop/widegotou.png')
puppet = img.copy()
h, w = img.shape[0], img.shape[1]
click_points = []

def get_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('origin', img)        
        click_points.append([x, y])
        if len(click_points) == 4:
            src_point = np.float32(click_points)
            dst_point = np.float32([[0, 0], [w, 0], [0, h], [w, h]])



            matrix = cv2.getPerspectiveTransform(src_point, dst_point)
            warped = cv2.warpPerspective(puppet, matrix, (w, h))

            cv2.namedWindow('warped', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('warped', warped)
            # cv2.imwrite('C:/Users/29560/Desktop/4.jpg', warped)
            cv2.waitKey(0)


cv2.namedWindow('origin', cv2.WINDOW_KEEPRATIO)
cv2.imshow('origin', img)
cv2.setMouseCallback('origin', get_points)
cv2.waitKey(0)