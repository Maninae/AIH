import numpy as np
import pickle
import os
# import sys
# sys.path.append('../utils')

from utils.image import *
from utils.debug import debug
from utils.data import *
from model.model import *
import tensorflow as tf

from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.metrics import mae, binary_accuracy
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


BATCH_SIZE = 32
NUM_EPOCHS = 30
CLASS_WEIGHTS = {0: 1., 1: 1.}   # To weight the rarer 1s more. customize on sensor in main

valid_train_sensor_ids = ['02', '04', '06', '08', '11', '15', \
                          '23', '39', '52', '59', '62', '63', '72', 'all']
model_base_name = "hhd_model_"

def start_new_training(sensor_id, data_path=None):
    print("Creating a new model and starting training clean.")
    model = create_new_model()
    model.summary()
    
    # TODO: Make recall and precision metrics for sensors that have skewed 0/1 distributions

    sgd = SGD(lr=0.001, decay=0.00001)
    model.compile(loss=binary_crossentropy,
                  optimizer=sgd,
                  metrics=[binary_accuracy, precision_metric, recall_metric])

    train(model, sensor_id, custom_path=data_path)



def resume_training(saved_model_path, sensor_id, data_path=None):
    print("Resuming training from saved model at %s" % saved_model_path)
    model = load_model(saved_model_path)
    model.summary()
    train(model, sensor_id, custom_path=data_path)



def train(model, sensor_id, custom_path=None):
    
    data_path = 'dataset/%s/' % sensor_id
    # Provided data_path overrides default data path (from local Git organization)
    if custom_path is not None:
        data_path = custom_path

    train_data, train_label, dev_data, dev_label = load_data(data_path)
    print("Running the fit call...")

    savepath = "model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    # For Floyd output:
    # os.makedirs("/output/model/%s/" % sensor_id)
    # savepath = "/output/model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    
    checkpointer = ModelCheckpoint(savepath, monitor='val_binary_accuracy')

    global CLASS_WEIGHTS
    print("Using class weights: %s" % str(CLASS_WEIGHTS))

    history = model.fit(x=train_data,
                        y=train_label,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(dev_data, dev_label),
                        shuffle=True,
                        class_weight=CLASS_WEIGHTS,
                        callbacks=[checkpointer])

    # Code to graph the history loss and metrics is in ```test.py```.
    print(history.history)

    # For Floyd output:
    # with open("/output/model/%s/history.pkl" % sensor_id, 'wb') as f:
    with open("model/%s/history.pkl" % sensor_id, 'wb') as f:
        pickle.dump(history.history, f)


def train_on_sensor(sensor_id, saved_model_path, custom_data_path):
    assert sensor_id in valid_train_sensor_ids
    
    # For saving the model later, make this dir
    sensordir = 'model/%s/' % sensor_id
    prev_exists = os.path.isdir(sensordir)
    if not prev_exists:
        os.makedirs(sensordir)
    
    if os.path.exists(saved_model_path):
        # Load existing model
        resume_training(saved_model_path, sensor_id, data_path=custom_data_path)
    else:
        # start training from scratch
        start_new_training(sensor_id, data_path=custom_data_path)


if __name__ == "__main__":
    sensor_id = prompt_for_sensor()
    #sensor_id = 'all'
    
    # FloydHub datasets are mounted at root - that's what this line was for
    # custom_data_path = "/dataset/all/"
    custom_data_path = None

    # Change weighting of positive samples in CLASS_WEIGHTS if desired
    CLASS_WEIGHTS[1] = 1.
    # CLASS_WEIGHTS[1] = 1., 1.7, 5., etc.

    
    # For retrieval of previous model
    #saved_model_path = ""
    saved_model_path = prompt_for_saved_model()

    # For saving the model later - we overwrote model_base_name when training the general model
    # model_base_name = "hhd_model__sensorALL_"
    
    train_on_sensor(sensor_id, saved_model_path, custom_data_path)

