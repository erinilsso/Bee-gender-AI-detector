import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import shutil

# 0. Create 'obj' folder in "data/" folder
# 1. Iterate over label files in label dirs
# 2. Find corresponding image file (one "step" up)
# 3. Move label file and image file into 'obj'

label_dirs = [
  'data/Beehive2_summer/labels/yolo', 
  'data/Raw data/Pictures Ericsson/Pictures/labels/yolo', 
  #'data/Raw data/Pictures LE/pictures_hive1/labels/yolo',
  'data/Raw data/Pictures LE/pictures_hive2/labels/yolo',
]

def getLabelsInDir(dir_path):
  label_list = []
  print(dir_path)
  for filename in glob.glob(dir_path + '/*.txt'):
    label_list.append(filename)
  return label_list

cwd = getcwd()

# Step 0
output_path =  cwd + '/data/obj'
if not os.path.exists(output_path):
  os.makedirs(output_path)

files = []
for label_dir in label_dirs:

  # Step 1
  label_paths = getLabelsInDir(label_dir)

  for label_path in label_paths:
    img_dir = os.path.dirname(os.path.dirname(os.path.dirname(label_path)))
    label_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Step 2
    for img_path in os.listdir(img_dir):
      if not os.path.isfile(img_dir + "/" + img_path) or img_path == '.DS_Store':
        continue
      img_name = os.path.splitext(os.path.basename(img_path))[0]
      if img_name == label_name:
        files.append((img_dir + "/" + img_path, label_path))

# Step 3
for f in files:
  img_path, label_path = f
  shutil.copy(img_path, output_path)
  shutil.copy(label_path, output_path)

print("Done :)")
