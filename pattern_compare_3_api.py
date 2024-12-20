from pattern_compare_3 import pattern_compare

"""
# mode = 0 （特征点）
         1 （图案比对）
         2 （图案差异）
         3 （颜色差异）
         4 （图案和颜色差异，分别为红色和黄色）
# 当mode = 0, 1时返回list为空
"""

alert_list = pattern_compare('C:/Users/29560/Desktop/1.jpg', 'C:/Users/29560/Desktop/3.jpg', mode=2, threshold=350)
# print(alert_list)