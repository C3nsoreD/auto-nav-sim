import numpy as np
import os
import cv2
import random
import pickle


DATADIR = "data"
CATEGORIES = ["not_safe", "safe"]
training_data = []

def dataset_gen():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) # Creates 0, 1 association for the classes
        for image in os.listdir(path):
            # Create an image array
            image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            training_data.append([image_array, class_num])
