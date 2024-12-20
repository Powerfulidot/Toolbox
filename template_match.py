'''
Description   : 基于灰度的模板匹配
Time          : 2024/11/12 16:09:02
Author        : 宽后藤 
'''

import cv2

target = cv2.imread('C:/Users/29560/Desktop/2.jpg')
template = cv2.imread('C:/Users/29560/Desktop/1.jpg')

height, width = template.shape[:2]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = cv2.TM_SQDIFF_NORMED

method = eval(str(method))
result = cv2.matchTemplate(target, template, method)
# cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#匹配值转换为字符串
#对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
#对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(target, top_left, bottom_right, (0, 0, 225), 2)

cv2.namedWindow('babaji', cv2.WINDOW_NORMAL)
cv2.imshow('babaji', target)
cv2.waitKey()
cv2.destroyAllWindows()