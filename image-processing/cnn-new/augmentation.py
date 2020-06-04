from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator

import glob
import os

IMAGE_SIZE = (70, 70)


def augmentation(image_path, save_to):
    f_image = load_img(image_path, target_size=IMAGE_SIZE)
    image_array = img_to_array(f_image)
    _image = expand_dims(image_array, axis=0)
    datagen = ImageDataGenerator(rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.15,
                                 zoom_range=0.5,
                                 channel_shift_range=10,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 )
    datagen.fit(_image)
    # print("__________________________________________")
    for x, val in zip(datagen.flow(_image, save_to_dir=save_to, save_prefix="img", save_format='png'), range(50)):
        pass


path = 'images/safe'
save_to = 'data/safe'
try:
    os.mkdir(save_to)
    for image in glob.glob((path + "/*.jpg")):
        print("Working on %s" % image)
        augmentation(image, save_to)
except OSError as e:
    print(e)

path = 'images/not_safe'
save_to = 'data/not_safe'


try:
    os.mkdir(save_to)
    for image in glob.glob((path + "/*.jpg")):
        print("Working on %s" % image)
        augmentation(image, save_to)
except OSError as e:
    print(e)
