import numpy as np
import os 
from matplotlib import pyplot as plt 
import cv2 
import random 
import pickle


file_list = []
class_lsit = []

DATADIR = "C:/Users/urgab/Desktop/git-repos/auto-nav-sim/image-processing/cnn-model/data"
CATEGORIES = ['not_safe_aug', 'safe_aug']

IMG_SIZE = 100
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([img_array, class_num])


create_training_data()
print(len(training_data))

random.shuffle(training_data)
X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X/255.0
print(X.shape[1:])


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
#
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
print(X[1])


