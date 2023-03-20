import cv2
from matplotlib import pyplot as plt
import getCaptcha

img = cv2.imread('big_img1.jpeg',0)#读入灰度图
img3 = cv2.imread('big_img1.jpeg',1)#读入彩色图

img2 = img.copy()
template = cv2.imread('small_img.jpeg',0)#读入小图
w,h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods :
    img = img2.copy()
    method = eval(meth)

    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #其中TM_SQDIFF 和TM_SQDIFF_NORMED 算法使用最小值优先法则
    if method in ['cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img3,top_left,bottom_right,(0,0,255),2)
    
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img3)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()


