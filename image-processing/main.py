
import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt 

"""
Templating matching using opencv 

"""
def main(image):
    # create an image copy
    img_copy = image.copy()
    template = cv2.imread('images/template.jpg', 0)
    w, h = template.shape[::-1]

    # Method for template matchin
    methods = 'cv2.TM_CCOEFF_NORMED'
    # _methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #         'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # method = eval(method)

    #applying template matching 
    method = eval(methods)

    res = cv2.matchTemplate(img_copy,template, method)
    min_val, max_val, min_loc, max_loc =  cv2.minMaxLoc(res)

    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img_copy, top_left, bottom_right, (0,255,0), 20)

    plt.subplot(121), plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_copy, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)

    plt.show()

if __name__ == "__main__":
    mypath = 'images/samples/image-00000.jpg'
    cv_image = cv2.imread(mypath, 0)
    main(cv_image)
    
    # ----------------------------------------------------------
    # onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f))]
    # images = np.empty(len(onlyfiles), dtype=object)
    # print(onlyfiles)
    # for i in range(0, len(onlyfiles)):

    #     images[i] = cv2.imread(str(os.path.join(mypath, onlyfiles[i])), 0)
    #     print(images[i])
    #     # cv_image = cv2.imread(images[i], 0)
        
    #     main(images[i])
    