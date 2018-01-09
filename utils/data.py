import os
import numpy as np

valid_sensor_ids = ['02', '04', '06', '08', '10', '11', '15', \
                    '21', '22', '23', '24', '39', '52', '59', \
                    '62', '63', '72']

def load_data(sensor_path):
    print("Loading data at: %s." % sensor_path)
    # Give it as "dataset/02/" with end slash
    sensor_id = sensor_path[-3:-1]
    assert sensor_id in valid_sensor_ids

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


