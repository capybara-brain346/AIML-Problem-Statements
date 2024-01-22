import tensorflow as tf
import os
import cv2
from PIL import Image
gpus = tf.config.experimental.list_physical_devices('CPU')

data_dir = 'data'
print(os.listdir(data_dir))
image_ext = ['jpeg', 'jpg', 'bmp', 'png']

DeprecationWarning: 'imghdr' is deprecated and slated for removal in Python 3.13
