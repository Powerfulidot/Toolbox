'''
Description   : 通过SIFT或SURF提取两张图片的特征点
Time          : 2024/10/30 11:31:54
Author        : 宽后藤 
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
 
img1 = cv2.imread(r'pattern_1.png')
img2 = cv2.imread(r'pattern_2.png')

#可以添加nfeatures参数限定点的个数 
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT.create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1 # kd树
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks = 50)   # or pass empty dictionary
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1, des2, k = 2)
 
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
 
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]
        
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)
 
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.figure(figsize=(20,20)); 
plt.imshow(img3, cmap='gray'), plt.title('Matched Result'), plt.axis('off')
plt.show()
