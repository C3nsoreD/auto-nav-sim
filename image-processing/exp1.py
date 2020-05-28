""" Image object 
"""
import cv2 as cv 
import numpy as np
import imutils 
import glob
import matplotlib.pyplot as plt 

def contains_template(template_image, visualize=False, image_path="images"):
    """
    Finds the location of the template in a given image.
    Arguments: template image, image path
    Returns an image with the location marked in a rectangle.
    """
    found = None
    (tH, tW) = template_image.shape[:2]

    for imagePath in glob.glob(image_path + "/*.jpg"):
        img = cv.imread(imagePath) # Read the image file 
        
        _gray = cv.cvtColor(_resize(img, scale_percent=30), cv.COLOR_BGR2GRAY) 
    
        for scale in np.linspace(0.2, 1.0, 30)[::-1]:
            # this loop scales the image by 30% while searching for the image template 
            resized = imutils.resize(_gray, width=int(_gray.shape[1] * scale))

            r = _gray.shape[1]/float(resized.shape[1])

            # Break the loop when the image is smaller than the template.
            # this will prevent and error on cv.matchTemplate
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break 
            
            _image = 0
            edged = cv.Canny(resized, 50, 100)
            result = cv.matchTemplate(edged, template_image, cv.TM_CCOEFF)
            # experiment
            # w, h = template.shape[::-1] 
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
            # (min_val, maxVal, min_loc, maxLoc) = cv.minMaxLoc(result)
            
            # top_left = min_loc
            # bottom_right = (top_left[0] + tW, top_left[1] + tH)

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
        cv.rectangle(_gray, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.imshow("Image", _gray)
        cv.waitKey(0)

def _resize(img, scale_percent=50):
    """
    Takes an image and resizes, default it reduces to 50% 
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    """ resize image"""
    image = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    
    return image

        
def canny_template(image):
    #converts image to gray 
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    scale_percent = 250 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    """ resize image"""
    image = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    # applies canny edge detection 
    template = cv.Canny(image, 50, 200)
    
    return template

def filtered(gray_image):
    blur = cv.GaussianBlur(gray_image, (11, 11), 0)
    thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=4)

def function1(image):
    """
    Attempting to reduce noise(image shine from)
    """
    # image = image_resize(img)
    new_image = np.zeros(image.shape, image.dtype)
    
    alpha = 1.0 # Simple contrast control
    beta = 50    # Simple brightness control
    # Initialize values
    print(' Basic Linear Transforms ')
    print('-------------------------')
    try:
        alpha = float(0.0) # float: 1.0-3.0
        beta = int(100) # 1-100
    except ValueError:
        print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    cv.imshow('Original Image', image)
    cv.imshow('New Image', new_image)
    # Wait until user press some key
    cv.waitKey()

if __name__ == "__main__":
    template = canny_template(cv.imread("13.jpg"))
    print(template.shape[:2])
    """
    Trying to reduce image shine to reduce errors 
    """
    # image = cv.imread("images/2.jpg")
    # resized = image_resize(image)
    # test_funt(resized)
    contains_template(template, visualize=True)
