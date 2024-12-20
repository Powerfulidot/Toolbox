'''
Description   : 获取图片某点颜色值并识别颜色
Time          : 2024/10/30 11:30:09
Author        : 宽后藤 
'''

import cv2

color_counter = [0] * 10
color_distance = []

color_table = ['棕色', '粉色', '橙色', '绿色', '蓝色', '淡粉色', '青色', '紫色', '红色', '黄色']

                #棕色         #粉色         #橙色         #绿色         #蓝色
hsv_table = [(15,73,216), (159,68,255), (16,114,255), (65,65,242), (104,73,253),    # 正面
               #淡粉色        #青色         #紫色          #红色          #黄色
             (168,38,254), (90,76,230), (132,55,229), (176,222,193), (24,127,255), 
             (12,93,208), (166,96,246), (16,140,255), (52,85,237), (105,112,229),   # 背面
             (171,41,236), (88,105,212), (129,74,222), (176,151,194), (25,158,237)]

lab_table = [(198,134,148), (209,162,114), (216,139,166), (231,99,146), (220,123,109),  #正面
             (235,141,125), (229,105,122), (204,143,107), (112,193,162), (230,126,179), 
             (179,140,151), (191,168,120), (205,145,174), (225,97,159), (184,126,97),   #背面
             (213,143,128), (205,100,123), (175,147,98), (120,175,144), (215,123,190)]

def calcDeltaE(x1, y1, z1, x2, y2, z2):
    deltaE = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return deltaE

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:      # 鼠标左击按下
        # 获取鼠标按下位置的颜色值
        a, b, c = hsv[y, x]
        # a, b, c = lab[y, x]

        # for color in hsv_table:          # 使用hsv色彩比较
        for color in lab_table:            # 使用lab色彩比较
            color_distance.append(calcDeltaE(a, b, c, color[0], color[1], color[2]))
        color_index = color_distance.index(min(color_distance))
        if color_index >= 10:
            color_index -= 10
        
        print(str(a),str(b),str(c), color_table[color_index])
        color_distance.clear()

 
img = cv2.imread(r'C:/Users/29560/Downloads/WeChat Files/wxid_z2p848y5xr2f22/FileStorage/File/2024-12/原子灰/yzh1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img', mouse_callback)       # 设置鼠标回调

cv2.imshow('img', img)
cv2.waitKey(0)