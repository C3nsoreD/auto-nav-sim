""" Image object 
"""
import cv2 as cv 
import numpy as np
import imutils 
import glob
import matplotlib.pyplot as plt 

def contains_template(template_image, visualize=False, image_path="images"):
    # 
    found = None
    (tH, tW) = template_image.shape[:2]

    for imagePath in glob.glob(image_path + "/*.jpg"):
        img = cv.imread(imagePath)
        
        scale_percent = 10 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        """ resize image"""
        image = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        _gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        for scale in np.linspace(0.2, 1.0, 50)[::-1]:

            resized = imutils.resize(_gray, width=int(_gray.shape[1] * scale))
            r = _gray.shape[1]/float(resized.shape[1])

            if resized.shape[0] < tH or resized.shape[1] < tW:
                break 

            edged = cv.Canny(resized, 50, 200)
            result = cv.matchTemplate(edged, template_image, cv.TM_CCOEFF)
            # experiment
            # w, h = template.shape[::-1] 
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
            # (min_val, maxVal, min_loc, maxLoc) = cv.minMaxLoc(result)
            
            # top_left = min_loc
            # bottom_right = (top_left[0] + w, top_left[1] + h)

            """ check to see if the iteration should be visualized"""
            if visualize:
                """ draw a bounding box around the detected region """
                clone = np.dstack([edged, edged, edged])
                cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv.imshow("Visualize", clone)
                cv.waitKey(0)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)


        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        # (startX, startY) = (int(maxLoc[0]*resized), int(maxLoc[1] * resized))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # (endX, endY) = (int((maxLoc[0] + tW) * resized), int((maxLoc[1] + tH) * resized))
        """draw a bounding box around the detected result and display the image """
        # cv.rectangle(image, top_left, bottom_right, (0,0,255), 20)
        cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.imshow("Image", image)
        cv.waitKey(0)


def canny_template(image):
    #converts image to gray 
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # applies canny edge detection 
    template = cv.Canny(gray, 50, 200)
    
    return template


if __name__ == "__main__":
    template = canny_template(cv.imread("11.jpg"))
    print(template.shape[:2])
    contains_template(template, visualize=True)
