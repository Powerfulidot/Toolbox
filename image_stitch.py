'''
Description   : cv2库中的stitcher类（用于全景图拼接、文档拼接）
Time          : 2024/10/30 11:30:35
Author        : 宽后藤 
'''

import cv2

img1 = cv2.imread('C:/Users/29560/Desktop/5.jpg')
img2 = cv2.imread('C:/Users/29560/Desktop/6.jpg')
img3 = cv2.imread('C:/Users/29560/Desktop/7.jpg')

stitcher = cv2.Stitcher.create(mode=cv2.Stitcher_PANORAMA)
stitcher.setWaveCorrection(True)
stitcher.setPanoConfidenceThresh(1)
imgs = [img1, img2, img3]

status, stitched = stitcher.stitch(imgs)
print(status)

if status == cv2.Stitcher_OK:
    cv2.namedWindow('stitched', cv2.WINDOW_NORMAL)
    cv2.imshow('stitched', stitched)
    cv2.waitKey(0)