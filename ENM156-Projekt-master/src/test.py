import os
from os.path import isfile, join
import math
from xml.dom import minidom

import tensorflow as tf
import pandas as pd
import numpy as np

""" 
x = np.zeros((1, 2, 3))
print(x.shape)

dir = 'data/Raw data/Pictures LE/pictures_hive2/labels'
dir, _ = os.path.split(dir)
print(dir)
"""

""" 
path = 'data/Raw data/Pictures LE/pictures_hive2/labels/ID_2_2021-05-24_13-22-18.xml'
dom = minidom.parse(path)
size = dom.getElementsByTagName('size')[0]
width = size.getElementsByTagName('width')[0].firstChild.nodeValue
height = size.getElementsByTagName('height')[0].firstChild.nodeValue

print(size)
print(width)
print(height)
"""

""" 
tensor4d = np.zeros((3, 7, 7, 18))
print(tensor4d[0].shape)
"""

print(tf.config.list_physical_devices('GPU'))
