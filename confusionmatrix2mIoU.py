'''
Description   : 将yolov5输出的混淆矩阵转化为mIoU的脚本
Time          : 2024/10/30 11:27:15
Author        : 宽后藤 
'''

import numpy as np

def miou(hist):
 
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mPA = np.nanmean(classAcc)
 
    return miou, mPA

matrix = np.array([[110, 0, 0],
                   [0, 130, 2],
                   [2, 0, 0]])

mIoU, mPA = miou(matrix)
print('mIoU =', mIoU, '\n', 'mPA =', mPA)