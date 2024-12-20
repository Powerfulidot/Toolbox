'''
Description   : 依靠txt的标签生成png格式的mask，单通道，掩膜区域灰度值等于类别
Time          : 2024/10/30 11:31:44
Author        : 宽后藤 
'''

import cv2
import os
import shutil
import json
import numpy as np


images_path = r'D:\yolov5-master\datasets\ncepu_dark_cars\images\val/'
labels_path = r'D:\yolov5-master\datasets\ncepu_dark_cars\labels\val/'
# save_images_path = r'C:\seg\1_images/'
save_labels_path = r'D:\yolov5-master\datasets\ncepu_dark_cars\masks\val/'
# save_visuals_path = r'D:\yolov5-master\datasets\ncepu_dark_cars\visualize\train/'


# 提取坐标
def txt2mask_new(img_x, img_y, line):
    # 处理每一行的内容
    data = line.split('\n')[0]
    label = data.split(" ")[0]
    d = data.split(' ', -1)
    data = []
    for i in range(1, int(len(d) / 2) + 1):
        data.append([img_y * float(d[2 * i - 1]), img_x * float(d[2 * i])])
    data.append(data[0]) # 起点与终点连接
    data = np.array(data, dtype=np.int32)
    return label, data

#标签选取
def mask_select(img_path,label_path):
    #访问txt格式标签文件
    with open(label_path, "r") as f1:
        new_lines = f1.readlines()
    f1.close()

    image = cv2.imread(img_path)  # 读取图片信息
    image_h = image.shape[0]
    image_w = image.shape[1]

    visual = np.zeros((image_h, image_w, 3), np.uint8)  # 生成一张与原图大小相同的值全为0的图
    save_label = np.zeros((image_h, image_w, 1), np.uint8)  # 生成一张与原图大小相同、channels为1的图片

    #txt标签处理
    for line in new_lines:
        label, data = txt2mask_new(image_h, image_w, line)
        cv2.fillPoly(visual, [data], color=(0, 0, 255)) #填充多边形，[data]:多边形顶点，围成的区域就是分割区域;color:多边形围成的区域值为(0, 0, 255)
        #生成road值为0，background为1的.png格式的语义分割标签
        cv2.fillPoly(save_label, [data], color=int(label)+1)

    #图像标签可视化
    alpha = 1
    beta = 0.5
    save_visual = cv2.addWeighted(image, alpha, visual, beta, 0) #融合图像和标签，设置显示比例
    # cv2.namedWindow('img_save', cv2.WINDOW_NORMAL)
    # cv2.imshow("img_save", save_visual)
    # key = cv2.waitKey(0)
    #进行可视化筛选，键盘点q退出，点s保存，点其他会到下一张图片。
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     return 0, image, save_label, save_visual
    # elif key == ord('s'):
    #     cv2.destroyAllWindows()
    return 1, image, save_label, save_visual
    # else:
    #     cv2.destroyAllWindows()
    #     return 2, image, save_label, save_visual


#查询图片列表
img_items = os.listdir(images_path)
#对图片进行排序，查看自己图片命名自定义设置
# img_items.sort(key=lambda x:( "_".join(x.split("_")[:-1]), int(x.split('_')[-1].split('.jpg')[0])))
# img_items.sort(key=lambda x: (x.split('_')[-3], int(x.split('_')[-2]), int(x.split('_')[-1].split('.jpg')[0])))
# img_items.sort(key=lambda x: ("_".join(x.split("_")[:-2]), int(x.split('_')[-2]), int(x.split('_')[-1].split('.jpg')[0])))

t = 0
for i in img_items:
    print(i)
    #图像和标签路径
    image_path = os.path.join(images_path, i)
    label_path = os.path.join(labels_path, i[:-4]+".txt")

    # save_image_path = save_images_path + '{}.jpg'.format(i[:-4])
    save_label_path = save_labels_path + '{}.png'.format(i[:-4])
    # save_visual_path = save_visuals_path + '{}.jpg'.format(i[:-4])

    try:
        k, save_image, save_label, save_visual = mask_select(image_path, label_path)
    except FileNotFoundError as e:
        continue
    if k == 0:
        break
    elif k == 1:
        # cv2.imwrite(save_image_path, save_image)
        cv2.imwrite(save_label_path, save_label)
        # cv2.imwrite(save_visual_path, save_visual)

print('done')