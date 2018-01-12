import os
from os import path
import numpy as np

valid_sensor_ids = ['02', '04', '06', '08', '10', '11', '15', \
                    '21', '22', '23', '24', '39', '52', '59', \
                    '62', '63', '72', 'all']

# Only for inference! For training use |load_data|.
def load_data_for_prediction(data_path):
    # at the location `$data_path` there should be two folders '0' and '1'
    #  with samples from each class that are 240x320 npz files, etc.
    
    print("Loading data for PREDICTION at: %s." % data_path)
    data = []
    label = []

    print("Loading samples from class 0")
    for filename in [a for a in os.listdir(path.join(data_path, '0')) if a[-4:] == '.npz']:
        raw = np.load(path.join(data_path, '0', filename))
        data.append(raw['x'])
        label.append(0)

    print("loading samples from class 1")
    for filename in [a for a in os.listdir(path.join(sensor_path, '1')) if a[-4:] == '.npz']:
        raw = np.load(path.join(data_path, '1', filename))
        data.append(raw['x'])
        label.append(1)

    print("Shuffling data...")
    data = np.array(data)
    label = np.array(label)
    pm = np.random.permutation(data.shape[0])
    data = data[pm]
    label = label[pm]

    return data, label


def load_data(sensor_path):
    print("Loading data for TRAINING at: %s." % sensor_path)
    # Give it as "dataset/02/" with end slash
    
    train_data = []
    train_label = []

    print("loading data train 0")
    for filename in [a for a in os.listdir(sensor_path + 'train/0') if a[-4:] == '.npz']:
        raw = np.load(sensor_path + 'train/0/' + filename)
        train_data.append(raw['x'])
        train_label.append(0)
    print("loading data train 1")
    for filename in [a for a in os.listdir(sensor_path + 'train/1') if a[-4:] == '.npz']:
        raw = np.load(sensor_path + 'train/1/' + filename)
        train_data.append(raw['x'])
        train_label.append(1)

    dev_data = []
    dev_label = []

    print('loading data dev 0')
    for filename in [a for a in os.listdir(sensor_path + 'dev/0') if a[-4:] == '.npz']:
        raw = np.load(sensor_path + 'dev/0/' + filename)
        dev_data.append(raw['x'])
        dev_label.append(0)
    print('loading data dev 1')
    for filename in [a for a in os.listdir(sensor_path + 'dev/1') if a[-4:] == '.npz']:
        raw = np.load(sensor_path + 'dev/1/' + filename)
        dev_data.append(raw['x'])
        dev_label.append(1)

    # Make ndarrays
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    dev_data = np.array(dev_data)
    dev_label = np.array(dev_label)

    # Shuffling will occur in the model.fit keras call, won't do it here
    # print("Shuffling data...")
    # train_pm = np.random.permutation(len(train_data))
    # dev_pm = np.random.permutation(len(dev_data))
    # train_data = train_data[train_pm]
    # train_label = train_label[train_pm]
    # dev_data = dev_data[dev_pm]
    # dev_label = dev_label[dev_pm]

    # Add a channel axis for convolution
    train_data = train_data[:,:,:,None]
    dev_data = dev_data[:,:,:,None]

    return train_data, train_label, dev_data, dev_label

def prompt_for_sensor():
    sensor_id = input("Sensor to train on (2 digits): ")
    if sensor_id not in valid_train_sensor_ids:
        raise Exception("Not a valid sensor id for training")
    return sensor_id

def prompt_for_saved_model():
    # User has to provide the full path
    name = input("Relative path to model to load? e.g. 'model/06/hhd_model...' (Enter to skip): ")
    if len(name) > 0 and not os.path.isfile(name):
        raise Exception("Not a valid path / nothing found at that location")
    return name

def prompt_for_inference_data_path():
    data_path = input("Relative path to the data to predict on?")
    if not os.path.isdir(data_path):
        raise Exception("Path wasn't found")

    if not os.path.isdir(os.path.join(data_path, '1')) or \
       not os.path.isdir(os.path.join(data_path, '0')):
        raise Exception("The input path doesn't have both 0/ and 1/ dirs inside")

    return data_path


