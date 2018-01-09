import numpy as np
from debug import debug
from data import *
from image import *
import os
import pickle

# A = np.load('../dataset/04/0/20160930_082811_796.npz')
# A = A['x']
# debug(A)

# train_data, train_label, dev_data, dev_label = load_data('../dataset/04/')
# debug(train_data)
# debug(train_label)
# debug(dev_data)
# debug(dev_label)

# display_by_name('../dataset/04/dev/0/20160930_082810_738.npz')

# Had to transcribe by hand... ended at epoch 22 instead of 30 so history didn't get saved
sensor_06_history = {
	'loss' :                [0.8153, 0.5327, 0.4431, 0.3856, 0.3430, 0.3120, 0.2783, 0.2536, 0.2393, 0.2161, 0.2069, 0.1810, 0.1812, 0.1567, 0.1478, 0.1360, 0.1310, 0.1227, 0.1162, 0.1095, 0.0977, 0.0908],
	'binary_accuracy' :     [0.7236, 0.8122, 0.8411, 0.8582, 0.8813, 0.9031, 0.9118, 0.9214, 0.9313, 0.9353, 0.9400, 0.9428, 0.9448, 0.9539, 0.9547, 0.9599, 0.9603, 0.9658, 0.9678, 0.9682, 0.9734, 0.9770],
	'val_loss' :            [0.4675, 0.3465, 0.3419, 0.3406, 0.2799, 0.2340, 0.2235, 0.2361, 0.1998, 0.2215, 0.1930, 0.1955, 0.1793, 0.1833, 0.1756, 0.1583, 0.1656, 0.1786, 0.2607, 0.1830, 0.1493, 0.1905],
	'val_binary_accuracy' : [0.7919, 0.8250, 0.8352, 0.8527, 0.8831, 0.9033, 0.9144, 0.9107, 0.9153, 0.9208, 0.9153, 0.9134, 0.9374, 0.9365, 0.9401, 0.9392, 0.9291, 0.9401, 0.9190, 0.9429, 0.9466, 0.9383]
}

with open('model/06/sensor_06_history_first22.pkl', 'wb') as f:
	pickle.dump(sensor_06_history, f)
