'''
Description   : 根据灰度判断图像属于低照度还是过度曝光
Time          : 2024/11/29 15:22:31
Author        : 宽后藤 
'''

import cv2

# img = cv2.imread('C:/Users/29560/Desktop/dark/IMG_20240530_201040.jpg')
img = cv2.imread('C:/Users/29560/Desktop/light/IMG_20241116_152330.jpg')
img_grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

average = cv2.mean(img_grayed)[0]

if average >= 127.5:
    judgement = '过度曝光:' + str(average)
else:
    judgement = '低照度' + str(average)

cv2.namedWindow(judgement, cv2.WINDOW_NORMAL)
cv2.imshow(judgement, img)

cv2.waitKey(0)
cv2.destroyAllWindows()


