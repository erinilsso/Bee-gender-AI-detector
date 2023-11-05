
import os
import csv
from xml.dom import minidom


label_dirs = ['data/Raw data/Pictures LE/pictures_hive2/labels']
classes = ['Worker/female', 'Drone/male', 'Unknown/bee']

with open('data.csv', 'w') as file:
  writer = csv.writer(file, delimiter=',')

  writer.writerow(['NOTE: This will not be used for training'])

  writer.writerow(['img', 'dir', 'num bees', 'num female', 'num male', 'num unknown'])

  for dir in label_dirs:
    for filename in os.listdir(dir):
      class_counts = dict.fromkeys(classes, 0)
      label_count = 0
      filepath = os.path.join(dir, filename)
      dom = minidom.parse(filepath)
      labels = dom.getElementsByTagName('name')
      for label in labels:
        class_counts[label.childNodes[0].nodeValue] += 1
        label_count += 1
      writer.writerow([
        filename,
        dir, 
        label_count,
        class_counts[classes[0]],
        class_counts[classes[1]],
        class_counts[classes[2]]
      ])


