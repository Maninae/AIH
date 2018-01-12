import numpy as np
from debug import debug
from data import *
from image import *
import os
import pickle

##############################

# Dumping ground for miscellaneous, one-time, quick-and-dirty code.

##############################


# A = np.load('../dataset/04/0/20160930_082811_796.npz')
# A = A['x']
# debug(A)

# train_data, train_label, dev_data, dev_label = load_data('../dataset/04/')
# debug(train_data)
# debug(train_label)
# debug(dev_data)
# debug(dev_label)

# display_by_name('../dataset/04/dev/0/20160930_082810_738.npz')

# Had to transcribe by hand for initial tests... ended at epoch 22 instead of 30 so history didn't get saved
# sensor_06_history = {
#   'loss' :                [0.8153, 0.5327, 0.4431, 0.3856, 0.3430, 0.3120, 0.2783, 0.2536, 0.2393, 0.2161, 0.2069, 0.1810, 0.1812, 0.1567, 0.1478, 0.1360, 0.1310, 0.1227, 0.1162, 0.1095, 0.0977, 0.0908, 0.0918],
#   'binary_accuracy' :     [0.7236, 0.8122, 0.8411, 0.8582, 0.8813, 0.9031, 0.9118, 0.9214, 0.9313, 0.9353, 0.9400, 0.9428, 0.9448, 0.9539, 0.9547, 0.9599, 0.9603, 0.9658, 0.9678, 0.9682, 0.9734, 0.9770, 0.9770],
#   'val_loss' :            [0.4675, 0.3465, 0.3419, 0.3406, 0.2799, 0.2340, 0.2235, 0.2361, 0.1998, 0.2215, 0.1930, 0.1955, 0.1793, 0.1833, 0.1756, 0.1583, 0.1656, 0.1786, 0.2607, 0.1830, 0.1493, 0.1905, 0.1467],
#   'val_binary_accuracy' : [0.7919, 0.8250, 0.8352, 0.8527, 0.8831, 0.9033, 0.9144, 0.9107, 0.9153, 0.9208, 0.9153, 0.9134, 0.9374, 0.9365, 0.9401, 0.9392, 0.9291, 0.9401, 0.9190, 0.9429, 0.9466, 0.9383, 0.9484]
# }

# with open('/output/stupid_test_output.pkl', 'wb') as f:
#   pickle.dump(sensor_06_history, f)

def graph(sensor_id, final_epoch):
    history_path = "model/completed/%s/history__sensor%s__ep0-%s.pkl" % (sensor_id, sensor_id, final_epoch)
    with open(history_path, 'rb') as f:
        history = np.load(f)

    save_dir = 'assets/figures/%s' % sensor_id
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    build_graphs_of_training_metrics(history, save_dir)

def combining_datasets(sensor_id):
    os.system("mv %s/train/0/* all/train/0" % sensor_id)
    os.system("mv %s/train/1/* all/train/1" % sensor_id)
    os.system("mv %s/dev/0/* all/dev/0" % sensor_id)
    os.system("mv %s/dev/1/* all/dev/1" % sensor_id)

# loss =  [0.8333, 0.4626, 0.3661, 0.3140, 0.2823, 0.2586, 0.2442, 0.2291, 0.2220, 0.2118, 0.2053, 0.2007, 0.1942, 0.1923, 0.1860, 0.1812, 0.1796, 0.1749, 0.1719, 0.1702, 0.1674, 0.1662, 0.1643, 0.1618, 0.1601]
# acc  =  [0.7972, 0.9145, 0.9403, 0.9529, 0.9606, 0.9654, 0.9693, 0.9716, 0.9734, 0.9753, 0.9765, 0.9780, 0.9790, 0.9794, 0.9807, 0.9812, 0.9820, 0.9831, 0.9834, 0.9836, 0.9841, 0.9844, 0.9851, 0.9854, 0.9857]
# vloss = [0.1991, 0.1569, 0.1325, 0.1283, 0.1402, 0.1264, 0.1220, 0.1238, 0.1245, 0.1202, 0.1164, 0.1165, 0.1162, 0.1154, 0.1163, 0.1156, 0.1153, 0.1180, 0.1144, 0.1162, 0.1156, 0.1142, 0.1140, 0.1157, 0.1151]
# vacc  = [0.9698, 0.9817, 0.9898, 0.9910, 0.9883, 0.9924, 0.9928, 0.9930, 0.9933, 0.9939, 0.9948, 0.9941, 0.9946, 0.9946, 0.9948, 0.9948, 0.9947, 0.9950, 0.9950, 0.9948, 0.9951, 0.9957, 0.9955, 0.9947, 0.9951]

# dct = {
#     'loss' : loss,
#     'binary_accuracy': acc,
#     'val_loss': vloss,
#     'val_binary_accuracy': vacc
# }
# with open("model/completed/all/history__sensorALL__ep0-24.pkl", 'wb') as f:
#     pickle.dump(dct, f)

if __name__ == "__main__":
    graph('all', '24')
