import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model

from utils.data import load_data_for_prediction, prompt_for_inference_data_path
from training import prompt_for_saved_model

if __name__ == "__main__":
    model_path = prompt_for_saved_model()
    data_path  = prompt_for_inference_data_path()
    data, label = load_data_for_prediction(data_path)

    pass
