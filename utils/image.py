import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def depth_map_to_image(depth_map):
    img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    return img

def display(data):
    #data = np.load('20170121_210810_614.npz')
    depth_map = data['x'].astype(np.float32)
    ocean = depth_map_to_image(depth_map)
    cv2.imshow("Image", ocean)
    cv2.waitKey(0)

def display_by_name(name):
    data = np.load(name)
    display(data)

# Usage e.g.
#   build_graphs_of_training_metrics(history, "assets/figures/06")
def build_graphs_of_training_metrics(history, save_dir):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['binary_accuracy']
    val_acc = history['val_binary_accuracy']

    num_epochs = len(loss)
    ep_axis = [a for a in range(1, num_epochs+1)]
    
    # Plot the losses
    plt.plot(ep_axis, loss, 'b-', label='training loss')
    plt.plot(ep_axis, val_loss, 'r-', label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Avg. cross entropy loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.clf()

    # Plot the bin. accuracy
    plt.plot(ep_axis, acc, 'c-', label='training accuracy')
    plt.plot(ep_axis, val_acc, 'm-', label='validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Avg. binary accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracies.png'))
    plt.clf()


