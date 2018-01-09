import numpy as np
import cv2
# import sys
# sys.path.append('../utils')

from utils.image import *
from utils.debug import debug
from utils.data import load_data

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.metrics import mae, binary_accuracy


from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def precision_metric(y_true, y_pred):
    return K.mean(y_pred) # dummy

def recall_metric(y_true, y_pred):
    return K.mean(y_pred) # dummy

# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
def swish_activation(x):
    pass # implement and experiment with Swish units if you have time

def create_new_model():
    model = Sequential()
    
    model.add(Conv2D(16, (5,5), padding='same', kernel_initializer='he_uniform', activation='relu', input_shape=(240, 320, 1)))
    model.add(Conv2D(16, (5,5), padding='same', kernel_initializer='he_uniform', activation='relu', input_shape=(240, 320, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # (120, 160, 16)

    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # (60, 80, 32)

    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # (30, 40, 64)

    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # (15, 20, 64)

    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(Conv2D(16, (3,3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    # (15, 20, 16)

    model.add(Flatten())    # (4800,)
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

