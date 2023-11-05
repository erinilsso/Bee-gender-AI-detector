import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

dirs = [
  'data/Raw data/Pictures Ericsson/Pictures/labels', 
  #'data/Raw data/Pictures LE/pictures_hive1/labels/yolo',
  'data/Raw data/Pictures LE/pictures_hive2/labels',
  'data/Beehive2_summer/labels', 
]
classes = ['Worker/female', 'Drone/male', 'Unknown/bee']

def getLabelsInDir(dir_path):
    label_list = []
    for filename in glob.glob(dir_path + '/*.xml'):
        label_list.append(filename)
    return label_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h) 

def convert_annotation(dir_path, output_path, label):
    basename = os.path.basename(label)
    basename_no_ext = os.path.splitext(basename)[0]

    print("Parsing: " + str(dir_path) + ", " + str(label))

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes: #or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

cwd = getcwd()

for dir_path in dirs:
    full_dir_path = cwd + '/' + dir_path
    output_path = full_dir_path +'/yolo/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    label_paths = getLabelsInDir(full_dir_path)

    for label in label_paths:
        convert_annotation(full_dir_path, output_path, label)

    print("Finished processing: " + dir_path)

