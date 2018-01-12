import numpy as np
import cv2
# import sys
# sys.path.append('../utils')

from utils.image import *
from utils.debug import debug
from utils.data import load_data

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.metrics import mae, binary_accuracy
from keras.regularizers import l2

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


l2_lambda_param = 0.0001


# Taken from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
# Batch wise and not global, but gives a good sense perhaps
def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
get_custom_objects().update({"precision_metric": precision_metric})


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
get_custom_objects().update({"recall_metric": recall_metric})


# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
def swish(x):
    return K.sigmoid(x) * x
get_custom_objects().update({'swish': Activation(swish)})


def create_new_model():
    model = Sequential()
    
    model.add(Conv2D(16, (5,5), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish, input_shape=(240, 320, 1)))
    model.add(Conv2D(16, (5,5), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.2))
    # (120, 160, 16)

    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.2))
    # (60, 80, 32)

    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.2))
    # (30, 40, 64)

    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.2))
    # (15, 20, 64)

    model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    model.add(Conv2D(16, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda_param), activation=swish))
    #model.add(Dropout(0.2))
    # (15, 20, 16)

    model.add(Flatten())    # (4800,)
    model.add(Dense(100, activation=swish, kernel_regularizer=l2(l2_lambda_param)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

