'''
Description   : 使用lsd检测直线（目前效果不如hough和fld）
Time          : 2024/10/30 11:28:31
Author        : 宽后藤 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

bus_img = cv2.imread('bus_2.jpg')
origin = bus_img.copy()
gray = cv2.cvtColor(bus_img, cv2.COLOR_BGR2GRAY)

lsd = cv2.createLineSegmentDetector(0, scale=1)
lines = lsd.detect(gray)

for line in lines[0]:
    x1 = int(round(line[0][0]))
    y1 = int(round(line[0][1]))    
    x2 = int(round(line[0][2]))
    y2 = int(round(line[0][3]))    
    
    if x2 - x1 == 0:
        angle = 90
    else:
        k = -(y2 - y1) / (x2 - x1)
        angle = np.arctan(k) * 57.29577

    if abs(angle - 90) < 10:
        cv2.line(bus_img, (x1, y1), (x2, y2), (0,255,0), 2)


# plt.figure()
# plt.subplot(2,1,1), plt.imshow(origin), plt.axis('off'), plt.title('origin')
# plt.subplot(2,1,2), plt.imshow(bus_img), plt.axis('off'), plt.title('bus')
# plt.show()
# cv2.imwrite('doted_boos.jpg', bus_img)
cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)
cv2.imshow('babaji', bus_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('done')