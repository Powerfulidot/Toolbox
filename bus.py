'''
Description   : 使用切片灰度检测缝隙
Time          : 2024/10/30 11:28:40
Author        : 宽后藤 
'''

import cv2
import numpy as np

bus_img = cv2.imread('boos.jpg')
gray = cv2.cvtColor(bus_img, cv2.COLOR_BGR2GRAY)
# _, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

detect_height = int(bus_img.shape[0] * 4 / 5)
detect_line = gray[detect_height, :].tolist()
# detect_line = gray[:, detect_height].tolist()


dark_index = []
for i, value in enumerate(detect_line):
    if value < 50:
        dark_index.append(i)

print(dark_index, '\n')

accumulator = 0
for i, index in enumerate(dark_index):
    try:
        if abs(dark_index[i] - dark_index[i + 1]) > 5:
            if accumulator > 3:
                print(dark_index[i])                
                del dark_index[i - accumulator : i + 1]

            accumulator = 0
        else:
            accumulator += 1
    except:
        pass

print(dark_index, '\n')

for index in dark_index:
    cv2.circle(bus_img, (index, detect_height), 2, (0, 255, 0), -1)

cv2.imwrite('doted_boos.jpg', bus_img)
# cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)
# cv2.imshow('babaji', bus_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in bus_img.shape[1]:
#     pass


# def how_long():
print('done')