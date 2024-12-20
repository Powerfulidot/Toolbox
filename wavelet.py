import pywt
import cv2
import numpy as np
import os

path = 'C:/Users/29560/Desktop/huh/'
save_path = 'C:/Users/29560/Desktop/huh/'

reform_channels = []
file_list = os.scandir(path)
for imgs in file_list:
    img = cv2.imread(os.path.join(path, imgs.name))
    channels = cv2.split(img)
    # img_b = img[:,:,0]
    # img_g = img[:,:,1]
    # img_r = img[:,:,2]
    for channel in channels:
        cA, (cH, cV, cD) = pywt.dwt2(channel, 'haar')

        cH = cv2.GaussianBlur(cH, (3, 3), 0)
        cV = cv2.GaussianBlur(cV, (3, 3), 0)
        cD = cv2.GaussianBlur(cD, (3, 3), 0)
        # cH = cv2.bilateralFilter(cH, 5, 150, 150)
        # cV = cv2.bilateralFilter(cV, 5, 150, 150)
        # cD = cv2.bilateralFilter(cD, 5, 150, 150)

        channel = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        reform_channels.append(channel/np.max(channel) * 255)

    img_reformed = cv2.merge(reform_channels)

    cv2.imwrite('C:/Users/29560/Desktop/huh/reformed.jpg', img_reformed)
    print(imgs.name)

print('done')