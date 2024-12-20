'''
Description   : 批量修改图片尺寸
Time          : 2024/11/21 10:48:23
Author        : 宽后藤 
'''

import cv2
import os

path = 'C:/Users/29560/Desktop/oi/'
save_path = 'C:/Users/29560/Desktop/oi/'

file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name))

    img = cv2.resize(img, (300, 400))

    cv2.imwrite(os.path.join(save_path, imgs.name), img)
    print(imgs.name)


print('done')