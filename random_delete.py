'''
Description   : 随机删除
Time          : 2024/10/30 11:32:49
Author        : 宽后藤 
'''

import random
import os

path = 'C:/Users/29560/Downloads/STNet_LPRNet-master/data/CCPD2019/ccpd_weather/'
file = os.listdir(path)
num = 9499      #删除的数量，不是剩的
selected = random.sample(file, num)

for i in selected:
    os.remove(path + i)

print('done')