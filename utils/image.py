import cv2
import numpy as np

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