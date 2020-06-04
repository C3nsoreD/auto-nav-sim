
from keras.preprocessing import image
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os

IMG_SIZE = (70, 70)

model = tf.keras.models.load_model('my_model.h5')
