
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np
import scipy.ndimage
import glob
      
def image_preprocessing(image_path, scale=30):
    img_file = load_img(image_path, target_size=(100, 100)) # target_size is smaller 
    data = img_to_array(img_file)
    image = expand_dims(data, 0)
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
        height_shift_range=0.1,shear_range=0.15, 
        zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)
    
    datagen.fit(image)
    print("## Saving images in %s " % (str(save_here)))
    for x, val in zip(datagen.flow(image, save_to_dir=save_here, save_prefix='aug',save_format='png'), range(10)):
        pass



if __name__ == "__main__":
    # Collect and process class A images 
    image_path = 'images/safe'
    save_here = 'safe_aug'
    for imagePath in glob.glob(image_path + "/*.jpg"):
        image_preprocessing(imagePath)
    # Collect and process the class B images 
    image_path = 'images/not_safe'
    save_here = 'not_safe_aug'
    for imagePath in glob.glob(image_path + "/*.jpg"):
        image_preprocessing(imagePath)
