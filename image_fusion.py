'''
Description   : 通过检测两张图片的共同特征点来拼接多张图片（只能是奇数张）
Time          : 2024/10/30 11:30:20
Author        : 宽后藤 
'''

import cv2
import numpy as np
import time

starttime=time.time()
img1 = cv2.imread('C:/Users/29560/Desktop/5.jpg')
img2 = cv2.imread('C:/Users/29560/Desktop/6.jpg')
img3 = cv2.imread('C:/Users/29560/Desktop/7.jpg')

# imgs = [img1, img2, img3]
# center_img = imgs[len(imgs) // 2]
# left_imgs = imgs[0 : len(imgs) // 2 + 1]
# right_imgs = imgs[len(imgs) // 2 : ]

# img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# surf = cv2.xfeatures2d.SURF_create(5000, nOctaves=4, extended=False, upright=True)
sift = cv2.SIFT.create()

#surf=cv2.xfeatures2d.SIFT_create()
# kp1, descrip1 = surf.detectAndCompute(img1, None)
# kp2, descrip2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)


flann = cv2.FlannBasedMatcher(indexParams, searchParams)

'''
右
'''
kp2, descrip2 = sift.detectAndCompute(img2, None)
kp3, descrip3 = sift.detectAndCompute(img3, None)
match = flann.knnMatch(descrip2, descrip3, k=2)


good=[]
for i, (m,n) in enumerate(match):
        if(m.distance < 0.75*n.distance):
                good.append(m)

if len(good) > 5:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    ano_pts = np.float32([kp3[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
    warp_right = cv2.warpPerspective(img3, np.linalg.inv(M), (img2.shape[1] + img3.shape[1], img3.shape[0]))
    direct = warp_right.copy()
    direct[0 : img2.shape[0], 0 : img2.shape[1]] = img2
    simple = time.time()

    rows,cols = img2.shape[:2]
    
    for col in range(0,cols):
        if img2[:, col].any() and warp_right[:, col].any():#开始重叠的最左端
            left = col
            break
    for col in range(cols-1, 0, -1):
        if img2[:, col].any() and warp_right[:, col].any():#重叠的最右一列
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not img2[row, col].any():#如果没有原图，用旋转的填充
                res[row, col] = warp_right[row, col]
            elif not warp_right[row, col].any():
                res[row, col] = img2[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img2[row, col] * (1-alpha) + warp_right[row, col] * alpha, 0, 255)

    warp_right[0 : img2.shape[0], 0 : img2.shape[1]] = res
    final = time.time()
    # direct = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)


    # cv2.namedWindow('simple', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('simple', direct)
    # cv2.waitKey(0)
    # plt.imshow(img3,), plt.show()
    # warpImg = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)


    # plt.imshow(img4,), plt.show()
    # print("simple stich cost %f"%(simple-starttime))
    # print("total cost %f"%(final - starttime))

    # cv2.imwrite("simplepanorma.png",direct)
    # cv2.imwrite("fusion.png", img4)
else:
    print("not enough matches!")

'''
左
'''

img2_flip = cv2.flip(img2, 1)
img1_flip = cv2.flip(img1, 1)

kp2f, descrip2f = sift.detectAndCompute(img2_flip, None)
kp1f, descrip1f = sift.detectAndCompute(img1_flip, None)
match = flann.knnMatch(descrip2f, descrip1f, k=2)


good=[]
for i, (m,n) in enumerate(match):
        if(m.distance < 0.75*n.distance):
                good.append(m)

if len(good) > 5:
    src_pts = np.float32([kp2f[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    ano_pts = np.float32([kp1f[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
    warp_left = cv2.warpPerspective(img1_flip, np.linalg.inv(M), (img2_flip.shape[1] + img1_flip.shape[1], img1_flip.shape[0]))
    direct = warp_left.copy()
    direct[0 : img2_flip.shape[0], 0 : img2_flip.shape[1]] = img2_flip
    simple = time.time()

    rows,cols = img2_flip.shape[:2]
    
    for col in range(0,cols):
        if img2_flip[:, col].any() and warp_left[:, col].any():#开始重叠的最左端
            left = col
            break
    for col in range(cols-1, 0, -1):
        if img2_flip[:, col].any() and warp_left[:, col].any():#重叠的最右一列
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not img2_flip[row, col].any():#如果没有原图，用旋转的填充
                res[row, col] = warp_left[row, col]
            elif not warp_left[row, col].any():
                res[row, col] = img2_flip[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img2_flip[row, col] * (1-alpha) + warp_left[row, col] * alpha, 0, 255)

    warp_left[0 : img2_flip.shape[0], 0 : img2_flip.shape[1]] = res
    warp_left = cv2.flip(warp_left, 1)
    final = time.time()
    # direct = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)


    # cv2.namedWindow('simple', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('simple', direct)
    # cv2.waitKey(0)
    # warpImg = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)

    cv2.namedWindow('right', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('right', warp_right)
    cv2.namedWindow('left', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('left', warp_left)
    cv2.waitKey(0)
    # print("simple stich cost %f"%(simple-starttime))
    # print("total cost %f"%(final - starttime))

    # cv2.imwrite("right.jpg",warp_right)
    # cv2.imwrite("left.jpg", warp_left)

else:
    print("not enough matches!")

'''
融合
'''

canvas_1 = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] + img3.shape[1], 3), dtype='uint8')
canvas_2 = canvas_1.copy()
canvas_3 = canvas_1.copy()
canvas_1[0 : img1.shape[0], 0 : warp_left.shape[1]] = warp_left
canvas_2[0 : img1.shape[0], -warp_right.shape[1]:] = warp_right
canvas_3[0 : img1.shape[0], -warp_right.shape[1]:warp_left.shape[1]] = img2

fusioned = cv2.addWeighted(canvas_1, 0.7, canvas_2, 0.7, 0)
# fusioned = cv2.subtract(fusioned, canvas_3)

fusion_gray = cv2.cvtColor(fusioned, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(fusion_gray, 0, 255, cv2.THRESH_BINARY)

cv2.namedWindow('bin', cv2.WINDOW_KEEPRATIO)
cv2.imshow('bin', binary)
cv2.waitKey(0)

cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(cnts[0])
# cv2.rectangle(fusioned, (x, y), (x + w, y + h), (255, 255, 255), 2)
# cv2.namedWindow('fusion', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('fusion', fusioned)
# cv2.waitKey(0)


fusion_cut = fusioned[y : y + h, x : x + w]


cv2.namedWindow('fusioned', cv2.WINDOW_KEEPRATIO)
cv2.imshow('fusioned', fusion_cut)
cv2.waitKey(0)