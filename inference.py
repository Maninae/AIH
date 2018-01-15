import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model

from utils.data import *
from model.model import *

INFR_BATCH_SIZE = 16

if __name__ == "__main__":
    
    # The path to the model that we want to use
    # model_path = "model/completed/all/hhd_model__sensorALL__ep22-vloss=0.1140-vbacc=0.9955.h5"
    model_path = prompt_for_saved_model()
    
    # This has to be a directory with subdirs "0" and "1" under it, containing samples in each of the classes
    # data_path = "dataset/imbalanced/39"
    data_path  = prompt_for_inference_data_path()


    ###
    model = load_model(model_path)
    print("Loaded model at %s." % model_path)

    print("Model has compiled metrics: %s" % ", ".join(model.metrics_names))
    
    infr_data, infr_label = load_data_for_inference(data_path)
    print("Loaded data at %s." % data_path)

    print("Evaluating.")
    results = model.evaluate(x=infr_data, y=infr_label, batch_size=INFR_BATCH_SIZE)

    for metric_name, value in zip(model.metrics_names, results):
        print("%s: %.6f" % (metric_name, value))
