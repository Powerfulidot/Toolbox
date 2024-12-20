'''
Description   : cv2库中的泊松融合
Time          : 2024/10/30 11:32:33
Author        : 宽后藤 
'''

import cv2

def get_center(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)

        # Seamlessly clone src into dst and put the results in output
        mixed_clone = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
        cv2.namedWindow('整蛊', cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('整蛊', get_center)
        cv2.imshow('整蛊', mixed_clone)
        cv2.waitKey(0)

# Read images : src image will be cloned into dst
src = cv2.imread('C:/Users/29560/Desktop/1.png')
dst = cv2.imread('C:/Users/29560/Desktop/2.jpg')
mask = cv2.imread('C:/Users/29560/Desktop/mask.png')

src = cv2.resize(src, (640, 320))
dst = cv2.resize(dst, (1920, 1080))
# mask = cv2.resize(mask, (int(src.shape[1]), int(src.shape[0])))
# mask = 255 * np.ones(src.shape, src.dtype)    # Create an all white mask

cv2.namedWindow('选取中心', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('选取中心', get_center)
cv2.imshow('选取中心', dst)
cv2.waitKey(0)