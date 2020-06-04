from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import pickle
import keras

import numpy as np
# features
X = pickle.load(open("X.pickle", "rb"))
# labels
y = pickle.load(open("y.pickle", "rb"))
y = np.asarray(y)  # y as numpy array

image_size = (100, 100)
batch_size = 32


def build_model(input_shape):
    model = Sequential()
    # inputs = keras.Input(shape=input_shape)
    model.add(Conv2D(32, (3, 3), strides=2, padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # # 2 hidden layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    # model.add(Activation("relu"))

    # The output layer with 13 neurons, for 13 classes
    # model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def main(_size, epochs):
    model = build_model(X.shape[1:])
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

    history = model.fit(X, y, batch_size=_size, epochs=epochs, validation_split=0.2)
    model.save_weights('my_model.h5')
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


main(16, 50)
