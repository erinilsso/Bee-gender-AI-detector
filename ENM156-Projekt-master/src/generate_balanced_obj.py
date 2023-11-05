import os
import shutil
import random
from PIL import Image

label_dirs = [
  'data/Raw data/Pictures Ericsson/Pictures/labels/yolo', 
  #'data/Raw data/Pictures LE/pictures_hive1/labels/yolo',
  'data/Raw data/Pictures LE/pictures_hive2/labels/yolo',
  'data/Beehive2_summer/labels/yolo', 
]

classes = ['Worker/female', 'Drone/male', 'Unknown/bee']

def getLabelPathsInDir(label_dir):
  label_list = []
  for filename in os.listdir(label_dir):
    if os.path.splitext(filename)[1] == ".txt":
      label_list.append(label_dir + "/" + filename)
  return label_list

def hasClass(label_path, class_index):
  if class_index >= len(classes):
    print("Class index " + str(class_index) + " out of range! (Max is " + str(len(classes)) + ")")
    return False
  with open(label_path) as label_file:
    for line in label_file.readlines():
      if int(line.split(' ')[0]) == class_index:
        return True
  return False

def img_path_from_label_path(label_path):
  label_name = os.path.splitext(os.path.basename(label_path))[0]
  img_dir = os.path.dirname(os.path.dirname(os.path.dirname(label_path)))
  for img_path in os.listdir(img_dir):
    if not os.path.isfile(img_dir + "/" + img_path) or img_path == '.DS_Store':
      continue
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if img_name == label_name:
      return img_dir + "/" + img_path
  print("I'm sorry, Dave")
  return ""

def augment(label_path, output_path):

  img_path = img_path_from_label_path(label_path)
  img_name = os.path.splitext(os.path.basename(img_path))[0]
  img_ext = os.path.splitext(os.path.basename(img_path))[1]
  
  # Flip image horizontally, vertically and both together
  original = Image.open(img_path)
  flipped_horz = original.transpose(method=Image.FLIP_LEFT_RIGHT)
  flipped_vert = original.transpose(method=Image.FLIP_TOP_BOTTOM)
  flipped_both = original.transpose(method=Image.ROTATE_180)
  flipped_horz.save(output_path + "/" + img_name + "_horz" + img_ext)
  flipped_vert.save(output_path + "/" + img_name + "_vert" + img_ext)
  flipped_both.save(output_path + "/" + img_name + "_both" + img_ext)
  original.close()
  flipped_horz.close()
  flipped_vert.close()
  flipped_both.close()

  label_lines_horz = []
  label_lines_vert = []
  label_lines_both = []

  with open(label_path) as label_file:
    for line in label_file.readlines():
      values = line.split(' ')
      label_lines_horz.append(values[0] + " " + str(1-float(values[1])) + " " + values[2] + " " + values[3] + " " + values[4])
      label_lines_vert.append(values[0] + " " + values[1] + " " + str(1-float(values[2])) + " " + values[3] + " " + values[4])
      label_lines_both.append(values[0] + " " + str(1-float(values[1])) + " " + str(1-float(values[2])) + " " + values[3] + " " + values[4])

  with open(output_path + "/" + img_name + "_horz.txt", 'w') as label_file_horz:
    label_file_horz.writelines(label_lines_horz)
  with open(output_path + "/" + img_name + "_vert.txt", 'w') as label_file_vert:
    label_file_vert.writelines(label_lines_vert)
  with open(output_path + "/" + img_name + "_both.txt", 'w') as label_file_both:
    label_file_both.writelines(label_lines_both)
  

# Generate output path
output_path =  'data/obj'
if not os.path.exists(output_path):
  print("Creating output path '" + output_path + "'")
  os.makedirs(output_path)

# Relative paths to label files including specific classes 
# Duplicates between them are allowed (and expected)
female_label_paths = []
male_label_paths = []
unknown_label_paths = []

# Identify which label files include which classes
for label_dir in label_dirs:

  label_paths = getLabelPathsInDir(label_dir)

  # Identify and categorize label paths
  for label_path in label_paths:
    if hasClass(label_path, 0):   # Worker/female
      female_label_paths.append(label_path)
    if hasClass(label_path, 1):   # Drone/male
      male_label_paths.append(label_path)
    if hasClass(label_path, 2):   # Unknown/bee
      unknown_label_paths.append(label_path)

print("Copying all images containing '" + classes[1] + "' to '" + output_path + "'")

# List of label paths (original; not in output) to use for training (i.e. put in output path)
used_label_paths = []

# Copy all males to output path
for male_label_path in male_label_paths:

  # Find corresponding image path
  img_path = img_path_from_label_path(male_label_path)

  shutil.copy(img_path, output_path)
  shutil.copy(male_label_path, output_path)

  used_label_paths.append(male_label_path)

print("Copying " + str(len(male_label_paths)) + " random images containing '" + classes[0] + "' to '" + output_path + "'")

# Copy equally many females to output path
for i in range(len(male_label_paths)):
  rand_index = random.randrange(len(female_label_paths))
  
  female_label_path = female_label_paths[rand_index]
  img_path = img_path_from_label_path(female_label_path)

  shutil.copy(img_path, output_path)
  shutil.copy(female_label_path, output_path)

  used_label_paths.append(female_label_path)

# Augmentation!
print("Augmenting all " + str(len(used_label_paths)) + " images...")
for label_path in used_label_paths:
  augment(label_path, output_path)

# Verify
total_class_counts = dict.fromkeys(classes, 0)
class_counts = dict.fromkeys(classes, 0)
for label_file in os.listdir(output_path):
  if os.path.splitext(label_file)[1] != '.txt':
    continue
  found_class = dict.fromkeys(classes, False)
  with open(output_path + "/" + label_file) as f:
    for line in f.readlines():
      parts = line.split(' ')
      class_index = int(parts[0])
      if not found_class[classes[class_index]]:
        class_counts[classes[class_index]] += 1
        found_class[classes[class_index]] = True
      total_class_counts[classes[class_index]] += 1
print(str(class_counts[classes[1]]) + " of " + str(4*len(used_label_paths)) + " images contain '" + str(classes[1]) + "'") # 4x due to augmentation
print("Files with at least one occurence:")
print(class_counts)
print("Total occurrences:")
print(total_class_counts)

print("\nDone :)")
