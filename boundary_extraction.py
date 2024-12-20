'''
Description   : 从ccpd数据集图片的名称中提取四个顶点的坐标，适用于语义分割
Time          : 2024/10/30 11:27:43
Author        : 宽后藤 
'''

import numpy as np
import os
import cv2

def imgname2bbox(images_path, labels_path):
    dirs = os.listdir(images_path)
    for image in dirs:
        image_name = image.split(".")[0]

        # 根据图像名分割标注
        _, _, box, points, label, brightness, blurriness = image_name.split('-')

        # --- 边界框信息
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]

        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        # points = points[-2:]+points[:2]

        # 图片信息
        image_path = f"{images_path}{image}"
        img = cv2.imread(image_path)
        with open(labels_path + image_name + ".txt", "w") as f:
            # xyxy
            x_rd, y_rd = points[0]
            x_ld, y_ld = points[1]
            x_lu, y_lu = points[2]
            x_ru, y_ru = points[3]
            x_rd_n = x_rd / img.shape[1]
            y_rd_n = y_rd / img.shape[0]
            x_ld_n = x_ld / img.shape[1]
            y_ld_n = y_ld / img.shape[0]
            x_lu_n = x_lu / img.shape[1]
            y_lu_n = y_lu / img.shape[0]
            x_ru_n = x_ru / img.shape[1]
            y_ru_n = y_ru / img.shape[0]
            # xyxy2xywh
            # x_min, y_min = box[0]
            # x_max, y_max = box[1]
            # x_center = (x_min + x_max) / 2 / img.shape[1]
            # y_center = (y_min + y_max) / 2 / img.shape[0]
            # width = (x_max - x_min) / img.shape[1]
            # height = (y_max - y_min) / img.shape[0]
            
            f.write(f"0 {x_rd_n:.6} {y_rd_n:.6} {x_ld_n:.6} {y_ld_n:.6} {x_lu_n:.6} {y_lu_n:.6} {x_ru_n:.6} {y_ru_n:.6}")


if __name__ == "__main__":
    images_train_path = "D:/yolov5-master/datasets/CCPD_3000_segment/images/train/"
    images_val_path = "D:/yolov5-master/datasets/CCPD_3000_segment/images/val/"
    labels_train_path = "D:/yolov5-master/datasets/CCPD_3000_segment/labels/train/"
    labels_val_path = "D:/yolov5-master/datasets/CCPD_3000_segment/labels/val/"

    # 从图片名字中提取ccpd的关键点信息
    dic_images = {0: images_train_path, 1: images_val_path}
    dic_labels = {0: labels_train_path, 1: labels_val_path}
    for i in dic_images:
        imgname2bbox(dic_images[i], dic_labels[i])
    print("done")