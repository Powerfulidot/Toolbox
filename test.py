import cv2
import numpy as np

img = cv2.imread('C:/Users/29560/Downloads/WeChat Files/wxid_z2p848y5xr2f22/FileStorage/File/2024-12/原子灰/yzh1.jpg')
# img = np.array(img, np.uint8)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

range = cv2.inRange(hsv_img, (0, 20, 50), (50, 255, 255))
mask = cv2.bitwise_and(img, img, mask=range)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.erode(mask, kernel_small, iterations=1)            # 先腐蚀再做闭运算
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

new_contours = []
for contour in contours:
    if cv2.contourArea(contour) < 20000 and cv2.contourArea(contour) > 1000:
        new_contours.append(contour)

cv2.drawContours(img, new_contours, -1, (0, 255, 0), 5)
text = 'number:' + str(len(new_contours))
cv2.putText(img, text, (50, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 8, (0, 0, 255), 10)

cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.namedWindow('sum', cv2.WINDOW_NORMAL)
cv2.imshow('origin', img)
cv2.imshow('sum', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
