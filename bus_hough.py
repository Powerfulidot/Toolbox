'''
Description   : 使用hough检测直线（目前效果最好）并用检测到的直线计算缝隙宽度
Time          : 2024/10/30 11:28:07
Author        : 宽后藤 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

bus_img = cv2.imread('C:/Users/29560/Downloads/WeChat Files/wxid_z2p848y5xr2f22/FileStorage/File/2024-11/门缝/IMG_20241112_144025.jpg')
# bus_img - cv2.resize(bus_img, (1920, 1080))

gray = cv2.cvtColor(bus_img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 150)
# _, bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(edge, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
# lines2 = cv2.HoughLines(edge, 1, np.pi/180, 100)
vertical_lines = []
horizontal_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    # x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 - x1 == 0:
        angle = 90
    else:
        k = abs((y2 - y1) / (x2 - x1))
        angle = np.arctan(k) * 57.29577

    if abs(angle - 90) < 20:
        vertical_lines.append([x1, y1, x2, y2])     # 获得垂直方向线段合集
    elif abs(angle) < 20:
        horizontal_lines.append([x1, y1, x2, y2])     # 获得水平方向线段合集


pure_vertical_lines = []
pure_horizontal_lines = []
pure_vertical_lines.append(vertical_lines[0])       # 初始化去重后的垂直方向线段合集
pure_horizontal_lines.append(horizontal_lines[0])       # 初始化去重后的水平方向线段合集

for vertical_line in vertical_lines:                # 去除互相间隔太近的垂直线段
    x1, x2 = vertical_line[0], vertical_line[2]
    skip = 0
    
    for compared_line in pure_vertical_lines:
        if vertical_line == compared_line:
            break
        else:
            x3, x4 = compared_line[0], compared_line[2]

        if abs((x1 + x2) / 2 - (x3 + x4) / 2) < 5:
            break
        else:
            skip += 1
    if skip == len(pure_vertical_lines):
        pure_vertical_lines.append(vertical_line)

for horizontal_line in horizontal_lines:                # 去除互相间隔太近的水平线段
    y1, y2 = horizontal_line[1], horizontal_line[3]
    skip = 0
    
    for compared_line in pure_horizontal_lines:
        if horizontal_line == compared_line:
            break
        else:
            y3, y4 = compared_line[1], compared_line[3]

        if abs((y1 + y2) / 2 - (y3 + y4) / 2) < 5:
            break
        else:
            skip += 1
    if skip == len(pure_horizontal_lines):
        pure_horizontal_lines.append(horizontal_line)


for pure_vertical_line in pure_vertical_lines:
    x1, y1, x2, y2 = pure_vertical_line[0], pure_vertical_line[1], pure_vertical_line[2], pure_vertical_line[3]
    cv2.line(bus_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    width = 1
    i = 1
    measure_point_x = int((x1 + x2) / 2)
    measure_point_y = int((y1 + y2) / 2)
    while gray[measure_point_y, measure_point_x - i] < 50:
        i += 1

    width += i
    i = 1

    while gray[measure_point_y, measure_point_x + i] < 50:
        i += 1

    width += i
    
    text = 'width=' + str(width)
    cv2.putText(bus_img, text, (measure_point_x, measure_point_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

for pure_horizontal_line in pure_horizontal_lines:
    x1, y1, x2, y2 = pure_horizontal_line[0], pure_horizontal_line[1], pure_horizontal_line[2], pure_horizontal_line[3]
    cv2.line(bus_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    width = 1
    i = 1
    measure_point_x = int((x1 + x2) / 2)
    measure_point_y = int((y1 + y2) / 2)
    while gray[measure_point_y - i, measure_point_x] < 50:
        i += 1

    width += i
    i = 1

    while gray[measure_point_y + i, measure_point_x] < 50:
        i += 1

    width += i
    
    text = 'width=' + str(width)
    cv2.putText(bus_img, text, (measure_point_x, measure_point_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)



# plt.figure()
# plt.subplot(2,1,1), plt.imshow(origin), plt.axis('off'), plt.title('origin')
# plt.subplot(2,1,2), plt.imshow(bus_img), plt.axis('off'), plt.title('bus')
# plt.show()

# cv2.imwrite('hough_bus.jpg', bus_img)
cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)
cv2.imshow('babaji', bus_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('done')