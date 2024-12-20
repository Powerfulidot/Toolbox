'''
Description   : 使用hsv颜色特征筛选车牌区域并矫正（适用于目标检测）（重制版）
Time          : 2024/10/30 11:33:09
Author        : 宽后藤 

# 2024.10.29: 优先尝试拟合四边形，如失败使用最小外接矩形
'''

import cv2
import numpy as np
from imutils import perspective

def rectify(path, type=1):
    img = cv2.imread(path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                          # hsv空间

    if type == 0:
        img_regioned = cv2.inRange(img_hsv, (100, 43, 46), (124, 255, 255)) # 蓝牌阈值
    elif type == 1:
        img_regioned = cv2.inRange(img_hsv, (0, 3, 116), (76, 211, 255))    # 绿牌阈值

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_regioned = cv2.erode(img_regioned, kernel_small, iterations=1)            # 先腐蚀再做闭运算
    img_regioned = cv2.morphologyEx(img_regioned, cv2.MORPH_CLOSE, kernel_big)

    contours, _ = cv2.findContours(img_regioned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # 寻找轮廓

    num = 0
    maxlength = 0
    for i in range(len(contours)):                                          # 找最大的轮廓
        if len(contours[i]) > maxlength:
            maxlength = len(contours[i])
            num = i
    contour = contours[num]


    para = 0.001
    print('尝试使用轮廓拟合四边形')
    peri = cv2.arcLength(contour, closed=True)                              # 尝试使用轮廓拟合四边形
    while(1):
        # print('尝试para =', para)
        box = cv2.approxPolyDP(contour, para * peri, closed=True)
        if len(box) > 4:
            # print(len(box))
            para += 0.001
        elif len(box) == 4:
            new_box = []
            print('拟合四边形成功')
            for coords in box:
                new_box.append(coords[0])
            box = np.int0(new_box)
            box = np.int0(perspective.order_points(box))

            break
        elif len(box) < 4:
            print('拟合四边形失败，使用最小外接矩形')
            four_poly = cv2.minAreaRect(contour)                # 拟合失败时使用最小外接矩形
            box = cv2.boxPoints(four_poly)
            box = np.int0(perspective.order_points(box))        # 得到矩形四个顶点坐标

            break

    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    coord_o = np.float32(box)
    coord_d = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
    matrix = cv2.getPerspectiveTransform(coord_o, coord_d)

    warped = cv2.warpPerspective(img, matrix, (300, 100))

    cv2.imshow('img', img)
    cv2.imshow('img_regioned', img_regioned)
    cv2.imshow('warped', warped)
    cv2.waitKey(0)

path = '4.jpg'
rectify(path, type=0)           # 0蓝牌，1绿牌