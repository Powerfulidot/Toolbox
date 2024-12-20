'''
Description   : 先通过hough transform检测图片中的图片，计算直线的倾斜角度并实现对图片的旋转，鲁棒性差
Time          : 2024/10/30 11:34:02
Author        : 宽后藤 
'''

import os
import cv2
import math
import random
import numpy as np
from scipy import ndimage

path = '1.png'

# 第一次矫正
img = cv2.imread(path)
img_size = img.shape
img_h = img_size[0]
img_w = img_size[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
#霍夫变换
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
for line in lines:
	rho,theta = line[0]
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	if x1 == x2 or y1 == y2:
		continue
	
	t = float(y2-y1)/(x2-x1)
	angle = math.degrees(math.atan(t))
	if angle > 45 or angle < -45:             # 竖向线条
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)
	else:
		cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1)
	


	# rotate_angle = angle
	
# print("旋转角: " + str(rotate_angle))
# rotate_img = ndimage.rotate(img, rotate_angle)

# #第二次矫正
# gray_2 = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
# edges_2 = cv2.Canny(gray_2, 50, 150, apertureSize = 3)
# #霍夫变换
# lines_2 = cv2.HoughLinesP(edges_2, 1, np.pi/180, 10, minLineLength=50, maxLineGap=1)
# for line in lines_2:
# 	x3, y3, x4, y4 = line[0]
# 	if x3 == x4 or y3 == y4:
# 		continue
# 	cv2.line(rotate_img, (x3,y3), (x4,y4), (0,255,0), 2)
# 	t = float(y2-y1)/(x2-x1)
# 	twist_angle = math.degrees(math.atan(t))


cv2.imshow("4090", edges)
cv2.imshow("babaji", img)
# cv2.imshow("lucitic", rotate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()