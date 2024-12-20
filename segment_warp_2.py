'''
Description   : 将yolov5分割输出的坐标进行多边形填充，然后取最小外接矩形（适用于语义分割）
Time          : 2024/10/30 11:34:39
Author        : 宽后藤 

# segment_warp.py的升级版
# 2024.10.29: 优先尝试拟合四边形，如失败使用最小外接矩形
'''

import cv2
import numpy as np
from imutils import perspective

img_dir = 'D:/yolov5-master/runs/predict-seg/exp8/1.jpg'
label_dir = 'D:/yolov5-master/runs/predict-seg/exp8/labels/1.txt'

img_o = cv2.imread(img_dir)
img_w, img_h = img_o.shape[1], img_o.shape[0]
with open(label_dir, "r") as f:
    label = f.read()
    all_coords = label.split(' ', 1)[1]  # 去掉第一个0
    coords = all_coords.split(' ')       # 所有坐标分开

xs = coords[0 : len(coords) : 2]         # 取x坐标
ys = coords[1 : len(coords) : 2]         # 取y坐标
xs = list(map(float, xs))            # 字符串转浮点数
ys = list(map(float, ys))

pts = []
for x in xs:
    pts.append([int(x * img_w), int(ys[xs.index(x)] * img_h)])
pts = np.array(pts)                 # pts为整形坐标阵列
pts = cv2.convexHull(pts, clockwise=True)     # convexhull可以选取最外围的点并按顺时针或逆时针排列

background = np.zeros(img_o.shape)              # 全黑的mask
cv2.fillPoly(background, [pts], (0, 0, 255))    # 用pts坐标填充多边形

print('尝试使用轮廓拟合四边形')
para = 0.001
peri = cv2.arcLength(pts, closed=True)
while(1):
    # print('尝试para =', para)
    box = cv2.approxPolyDP(pts, para * peri, closed=True)
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
        four_poly = cv2.minAreaRect(pts)                # 拟合失败时使用最小外接矩形
        box = cv2.boxPoints(four_poly)
        box = np.int0(perspective.order_points(box))        # 得到矩形四个顶点坐标

        break

cv2.drawContours(background, [box], 0, (0, 255, 0), 2)

coord_o = np.float32(box)
coord_d = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
matrix = cv2.getPerspectiveTransform(coord_o, coord_d)

warped = cv2.warpPerspective(img_o, matrix, (300, 100))

cv2.namedWindow('origin', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('warped', cv2.WINDOW_KEEPRATIO)
cv2.imshow('origin', img_o)
cv2.imshow('mask', background)
cv2.imshow('warped', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
