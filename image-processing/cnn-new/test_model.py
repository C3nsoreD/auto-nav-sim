
from keras.preprocessing import image

from matplotlib import pyplot as plt
import numpy as np
from . import cnn-mode

IMG_SIZE = (70, 70)

model = tf.keras.models.load_model('my_model.h5')
img = tf.keras.preprocessing.image.load_img(
    "images/safe/3.jpg", target_size=IMG_SIZE
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)