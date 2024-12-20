'''
Description   : 读取预测区域坐标，提取边缘点并拉伸，适用于语义分割
Time          : 2024/10/30 11:35:25
Author        : 宽后藤 

# yolov5分割结果输出的坐标并不连续且不全（有时并不会包含顶点坐标），即使启用了retina-mask也一样
# ↑不连续的问题可以使用convexhull解决，但不全无法解决，优化方法见segment_warp_2.py
'''

import os
import cv2
import numpy as np

img_dir = 'D:/yolov5-master/runs/predict-seg/exp13/1.jpg'
label_dir = 'D:/yolov5-master/runs/predict-seg/exp13/labels/1.txt'

img_o = cv2.imread(img_dir)
img_w, img_h = img_o.shape[1], img_o.shape[0]
with open(label_dir, "r") as f:
    label = f.read()
    all_coords = label.split(' ', 1)[1]  # 去掉第一个0
    coords = all_coords.split(' ')       # 所有坐标分开
xs = coords[0 : len(coords) : 2]         # 取x坐标
ys = coords[1 : len(coords) : 2]         # 取y坐标
# xs = [i * img_w for i in xs]             # 反归一化
# ys = [i * img_h for i in ys]
xs = list(map(float, xs))            # 字符串转浮点数
ys = list(map(float, ys))
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

point_1 = [xs[ys.index(y_min)] * img_w, y_min * img_h]    # 找出边缘坐标
point_2 = [x_max * img_w, ys[xs.index(x_max)] * img_h]
point_3 = [xs[ys.index(y_max)] * img_w, y_max * img_h]
point_4 = [x_min * img_w, ys[xs.index(x_min)] * img_h]

coord_o = np.float32([point_1, point_2, point_3, point_4])
cv2.circle(img_o, (int(point_1[0]), int(point_1[1])), 5, (0, 0, 255), -1)
cv2.circle(img_o, (int(point_2[0]), int(point_2[1])), 5, (0, 255, 0), -1)
cv2.circle(img_o, (int(point_3[0]), int(point_3[1])), 5, (255, 0, 0), -1)
cv2.circle(img_o, (int(point_4[0]), int(point_4[1])), 5, (255, 255, 255), -1)


# coord_d = np.float32([[0, 0], [(x_max - x_min) * img_w, 0], 
#                       [(x_max - x_min) * img_w, (y_max - y_min) * img_h], 
#                       [0, (y_max - y_min) * img_h]])
coord_d = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
matrix = cv2.getPerspectiveTransform(coord_o, coord_d)
warped = cv2.warpPerspective(img_o, matrix, (300, 100))

# cv2.imwrite(os.path.join('C:/Users/29560/Desktop/', '1.jpg'), warped)

cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.imshow('origin', img_o)
cv2.imshow('warped', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
