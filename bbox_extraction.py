'''
Description   : 从ccpd数据集图片的名称中提取bbox坐标（xywh），适用于目标检测
Time          : 2024/10/30 11:27:29
Author        : 宽后藤 
'''

import numpy as np
import os
import cv2

def imgname2bbox(images_path, labels_path):
    dirs = os.listdir(images_path)
    for image in dirs:
        image_name = image.split(".")[0]
        box = image_name.split("-")[2]
        # 边界框信息
        box = box.split("_")
        box = [list(map(int, i.split('&'))) for i in box]
        # 图片信息
        image_path = f"{images_path}{image}"
        img = cv2.imread(image_path)
        with open(labels_path + image_name + ".txt", "w") as f:
            x_min, y_min = box[0]
            x_max, y_max = box[1]
            x_center = (x_min + x_max) / 2 / img.shape[1]
            y_center = (y_min + y_max) / 2 / img.shape[0]
            width = (x_max - x_min) / img.shape[1]
            height = (y_max - y_min) / img.shape[0]
            f.write(f"0 {x_center:.6} {y_center:.6} {width:.6} {height:.6}")


if __name__ == "__main__":
    images_train_path = "D:/yolov5-master/datasets/CCPD_enhanced/images/train/"
    images_val_path = "D:/yolov5-master/datasets/CCPD_enhanced/images/val/"
    labels_train_path = "D:/yolov5-master/datasets/CCPD_enhanced/labels/train/"
    labels_val_path = "D:/yolov5-master/datasets/CCPD_enhanced/labels/val/"

    # 从图片名字中提取ccpd的边界框信息，即(c, x, y, w, h)
    dic_images = {0: images_train_path, 1: images_val_path}
    dic_labels = {0: labels_train_path, 1: labels_val_path}
    for i in dic_images:
        imgname2bbox(dic_images[i], dic_labels[i])
    print("done")