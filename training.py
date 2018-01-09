import numpy as np
import cv2
import pickle
import os
# import sys
# sys.path.append('../utils')

from utils.image import *
from utils.debug import debug
from utils.data import load_data
from model.model import create_new_model, precision_metric
import tensorflow as tf

from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.metrics import mae, binary_accuracy
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


BATCH_SIZE = 32
NUM_EPOCHS = 30
CLASS_WEIGHTS = {0: 1., 1: 2.}   # To weight the rarer 1s more. customize on sensor

valid_train_sensor_ids = ['02', '04', '06', '08', '11', \
                          '23', '52', '62', '63', '72']
model_base_name = "hhd_model"

def start_new_training(sensor_id):
    print("Creating a new model and starting training clean.")
    model = create_new_model()
    model.summary()
    
    sgd = SGD(lr=0.001)
    model.compile(loss=binary_crossentropy,
                  optimizer=sgd,
                  metrics=[binary_accuracy])
    train(model, sensor_id)

def resume_training(modelpath, sensor_id):
    print("Resuming training from saved model at %s" % modelpath)
    model = load_model(modelpath)
    model.summary()
    train(model, sensor_id)

def train(model, sensor_id):
    train_data, train_label, dev_data, dev_label = load_data('dataset/%s/' % sensor_id)
    print("Running the fit call...")

    savepath = "model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    checkpointer = ModelCheckpoint(savepath, monitor='val_binary_accuracy')
    # Add callback for history

    history = model.fit(x=train_data,
                        y=train_label,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(dev_data, dev_label),
                        shuffle=True,
                        class_weight=CLASS_WEIGHTS,
                        callbacks=[checkpointer])

    # Do something with the history
    print(history.history)
    with open("model/%s/history.pkl" % sensor_id, 'wb') as f:
        pickle.dump(history.history, f)


def train_on_sensor(sensor_id, saved_model_name):
    assert sensor_id in valid_train_sensor_ids
    
    # For saving the model later
    sensordir = 'model/%s/' % sensor_id
    prev_exists = os.path.isdir(sensordir)
    if not prev_exists:
        os.makedirs(sensordir)
    
    if os.path.exists(sensordir + saved_model_name):
        # Load existing model
        resume_training(sensordir + saved_model_name, sensor_id)
    else:
        # start training from scratch
        start_new_training(sensor_id)


def prompt_for_sensor():
    sensor_id = input("Sensor to train on (2 digits): ")
    if sensor_id not in valid_train_sensor_ids:
        raise ValueError("Not a valid sensor id for training")
    return sensor_id

def prompt_for_saved_model():
    name = input("path to model to load? (Enter to skip): ")
    if len(name) > 0:
        assert os.path.exists(name)
    return name

if __name__ == "__main__":
    #sensor_id = prompt_for_sensor()
    sensor_id = '06'

    #saved_model_name = prompt_for_saved_model()
    saved_model_name = "hhd_model__22-0.191.h5"
    
    train_on_sensor(sensor_id, saved_model_name)

