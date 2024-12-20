'''
Description   : 测量宽度
Time          : 2024/11/12 16:08:28
Author        : 宽后藤 
'''

import cv2

img = cv2.imread('C:/Users/29560/Downloads/WeChat Files/wxid_z2p848y5xr2f22/FileStorage/File/2024-11/门缝/IMG_20241112_144025.jpg')
height, width = img.shape[0], img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]

cv2.namedWindow('lucitic', cv2.WINDOW_NORMAL)
cv2.imshow('lucitic', gray)
cv2.waitKey(0)

gap_ing = False
gap_vertical = []
gap_horizontal = []
for i in range(0, height):
    if gray[i, int(width / 2)] < 50 and gap_ing == False:
        gap_ing = True
        gap1_start = i
    elif gray[i, int(width / 2)] >= 50 and gap_ing == True:
        gap_ing = False
        gap1_end = i
        gap_vertical.append([gap1_start, gap1_end])

gap_ing = False
for i in range(0, width):
    if gray[int(height / 2), i] < 50 and gap_ing == False:
        gap_ing = True
        gap1_start = i
    elif gray[int(height / 2), i] >= 50 and gap_ing == True:
        gap_ing = False
        gap1_end = i
        gap_horizontal.append([gap1_start, gap1_end])


cv2.line(img, (int(width / 2), gap_vertical[0][0]), (int(width / 2), gap_vertical[0][1]), (0, 0, 255), 2)
cv2.putText(img, str(gap_vertical[0][1] - gap_vertical[0][0]), (int(width / 2), int((gap_vertical[0][0] + gap_vertical[0][1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.line(img, (int(width / 2), gap_vertical[1][0]), (int(width / 2), gap_vertical[1][1]), (0, 0, 255), 2)
cv2.putText(img, str(gap_vertical[1][1] - gap_vertical[1][0]), (int(width / 2), int((gap_vertical[1][0] + gap_vertical[1][1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.line(img, (int(width / 2), gap_vertical[0][1]), (int(width / 2), gap_vertical[1][0]), (0, 255, 0), 2)
cv2.putText(img, str(gap_vertical[1][0] - gap_vertical[0][1]), (int(width / 2), int((gap_vertical[1][0] + gap_vertical[0][1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.line(img, (gap_horizontal[0][0], int(height / 2)), (gap_horizontal[0][1], int(height / 2)), (0, 0, 255), 2)
cv2.putText(img, str(gap_horizontal[0][1] - gap_horizontal[0][0]), (int((gap_horizontal[0][0] + gap_horizontal[0][1]) / 2), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.line(img, (gap_horizontal[1][0], int(height / 2)), (gap_horizontal[1][1], int(height / 2)), (0, 0, 255), 2)
cv2.putText(img, str(gap_horizontal[1][1] - gap_horizontal[1][0]), (int((gap_horizontal[1][0] + gap_horizontal[1][1]) / 2), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.line(img, (gap_horizontal[0][1], int(height / 2)), (gap_horizontal[1][0], int(height / 2)), (255, 0, 0), 2)
cv2.putText(img, str(gap_horizontal[1][0] - gap_horizontal[0][1]), (int((gap_horizontal[1][0] + gap_horizontal[0][1]) / 2), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)


# cv2.imwrite('gap.jpg', img)
cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)
cv2.imshow('babaji', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('done')