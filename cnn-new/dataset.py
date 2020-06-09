import numpy as np
import os
import cv2
import random
import pickle

abs_path = "C:/Users/urgab/Desktop/git-repos/auto-nav-sim/image-processing/cnn-new/"
DATADIR = "data"
CATEGORIES = ["not_safe", "safe"]
training_data = []
image_size = 70

def dataset_gen():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # Creates 0, 1 association for the classes
        for image in os.listdir(path):
            # Create an image array
            image_path = os.path.join(path, image)
            image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            training_data.append([image_array, class_num])


def main():
    dataset_gen()
    random.shuffle(training_data)
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, image_size, image_size, 1)
    X = X/255.0

    # Save to pickle file
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


main()