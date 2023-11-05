
import os
import shutil
import random
from PIL import Image

obj_path = "/content/data/obj"
train_path = "/content/data/train.txt"
test_path = "/content/data/test.txt"

def augment(img_path, label_path, output_path):

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
  

# Find and store image paths (don't want to create new files while we're iterating over them)
images = []
for img_path in os.listdir(obj_path):
  if os.path.splitext(img_path)[1] == '.jpeg' or os.path.splitext(img_path)[1] == '.jpg':
    images.append(obj_path + "/" + img_path)
print(images)

# Augment all images (both train and test)
print("Augmenting images...")
for (i, img_path) in enumerate(images):
  if i % 20 == 0:
    print(str(i) + ": " + img_path)

  img_name = os.path.splitext(os.path.basename(img_path))[0]
  label_path = os.path.dirname(img_path) + "/" + img_name + ".txt"
  augment(img_path, label_path, obj_path)


# Update train.txt
print("Updating 'train.txt'...")
train_count = 0
line_count = 0
with open(train_path, "r+") as train_file:
  img_paths = []
  for img_path in train_file.readlines():
    line_count += 1
    img_paths.append(img_path)
  for img_path in img_paths:
    img_base = os.path.splitext(img_path)[0]
    img_ext = os.path.splitext(img_path)[1]
    train_file.write(img_base + "_horz" + img_ext)
    train_file.write(img_base + "_vert" + img_ext)
    train_file.write(img_base + "_both" + img_ext)
    train_count += 4

print(line_count)
line_count = 0

# Update test.txt
print("Updating 'test.txt'...")
test_count = 0
with open(test_path, "r+") as test_file:
  img_paths = []
  for img_path in test_file.readlines():
    line_count += 1
    img_paths.append(img_path)
  for img_path in img_paths:
    img_base = os.path.splitext(img_path)[0]
    img_ext = os.path.splitext(img_path)[1]
    test_file.write(img_base + "_horz" + img_ext)
    test_file.write(img_base + "_vert" + img_ext)
    test_file.write(img_base + "_both" + img_ext)
    test_count += 4
print(line_count)

tot = train_count + test_count
print("After augmentation:")
print("Training on " + str(train_count) + " of " + str(tot) + " images")
print("Testing on " + str(test_count) + " of " + str(tot) + " images")
