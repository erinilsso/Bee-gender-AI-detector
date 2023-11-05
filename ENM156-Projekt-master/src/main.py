
import os
from os.path import isfile, join
import math
from xml.dom import minidom

import tensorflow as tf
import pandas as pd
import numpy as np

print(tf.__version__)

# TODO:
# [x] Make sure there are BATCH_SIZE number of elements
# [x] Make sure a new batch is used each time
# [ ] Augment data
# [ ] Shuffle samples
# [ ] Decrease model size

classes = ['Worker/female', 'Drone/male', 'Unknown/bee']
label_dirs = ['data/Raw data/Pictures LE/pictures_hive2/labels']
checkpoint_path = 'checkpoints/checkpoint'

eps = 10**(-2)
print(eps)

# Hyper-parameters
EPOCHS = 3
BATCH_SIZE = 32
LR = 0.0001
SAVE_INTERVAL = BATCH_SIZE*10    # In # of samples

# NOTE:
# Images are either 1280x720x3 or 590x460x3.
# Both will have to be scaled by the first layer to fit the network's input shape

# Input image shape
img_width  = 448
img_height = 448
img_depth  = 3

# YOLOv1 hyper-parameters
S = 7
B = 3
l_coord = 5
l_noobj = 0.5
C = len(classes)

cell_w = img_width / S
cell_h = img_height / S

# YOLO box:  [x, y, sqrt w, sqrt h, i]
# Label box: [xmin, ymin, xmax, ymax]

# NOTE:
# Each cell's output is:
# [c1, x1, y1, sqrt w1, sqrt h1, c2, x2, y2, sqrt w2, sqrt h2, ..., p(1), p(2), p(3), ...]

# c: objectness score
# x: bounding box x coordinate (normalized within cell bounds)
# y: bounding box y coordinate (normalized within cell bounds)
# sqrt w: sqrt of bounding box width (normalized within image width)
# sqrt h: sqrt of bounding box height (normalized within image height)
# p(n): probability of cell containing class n

# NOTE: 
# In YOLOv1, each cell predicts multiple (B) bounding boxes, but only one class

# NOTE:
# I believe the two parameters are tensors, so we shouldn't need
# any more parameters
# NOTE:
# With a custom loss function, y_true technically doesn't need to have the 
# same dims as y_pred, but I guess it will--though, it requires me to format the target
# the same way as the outputs from the network.
# NOTE:
# It's basically sum-squared error everywhere, which is nice for gradient calculations.
# Hmm. But I don't think we will perform those calculations anyway... because autograd?
# NOTE:
# Ground truth tensor should only have one bounding box per cell (maximum), corresponding
# to the class distribution (which should be one-hot). The other box predictor values should
# be zeroed.
def YOLOv1_loss(y_true, y_pred):

  # i: cell index
  # j: bounding box predictor index

  # Distance, dimensions, objectness and classification loss
  dist_loss = 0
  dim_loss = 0
  obj_loss = 0
  class_loss = 0
  for b in range(BATCH_SIZE):
    for i in range(S*S):

      # Extract predictions
      objectness_scores = extract_cell_objectness_scores(y_pred[b], i)
      pred_boxes = list(map(lambda x: from_YOLO_box(*x), extract_cell_bounding_boxes(y_pred[b], i)))
      class_preds = extract_cell_class_probabilities(y_pred[b], i)

      # Skip if no object is in the cell (NOTE: Is this to-spec? I think so...)
      has_object = extract_cell_objectness_scores(y_true[b], i)[0] > eps
      if not has_object:
        # Penalize objectness using l_noobj
        # NOTE:
        # Not sure what 1_ij^noobj means.
        # There's no single box predictor responsible for the non-existent object (obviously)
        # so I'm assuming it means we do it for all box predictors, but only when there is
        # no object in the cell.
        for j in range(B):
          obj_loss += objectness_scores[j]**2   # No need for "-0"
        obj_loss *= l_noobj
        continue

      true_box = from_YOLO_box(*extract_cell_bounding_boxes(y_true[b], i)[0])
      true_class = extract_cell_class_probabilities(y_true[b], i)

      # Find box predictor responsible for this object
      best_match_idx = 0
      best_match = 0
      for j in range(B):
        match = IoU(*pred_boxes[j], *true_box)
        if j == 0 or match > best_match:
          best_match_idx = j
          best_match = match
      
      # Convert true_box back to YOLO form before calculating any loss
      true_box = to_YOLO_box(*true_box)

      # Box predictor at best_match_idx is now responsible for this object
      # -> Compare its coords, dimensions and objectness to ground truth 
      pred_box = to_YOLO_box(*pred_boxes[best_match_idx])
      objectness = objectness_scores[best_match_idx]

      dist_loss += l_coord * ((pred_box[0] - true_box[0])**2 + (pred_box[1] - true_box[1])**2) 
      dim_loss  += l_coord * ((pred_box[2] - true_box[2])**2 + (pred_box[3] - true_box[3])**2)
      obj_loss  += (objectness - 1)**2  # No need to fetch ground-truth objectness since it will be 1
      
      # Calculate class prediction loss
      # Independent of responsible box predictor, but makes sense to put it here anyway
      for (i, c) in enumerate(class_preds):
        class_loss += (c - true_class[i])**2

  return dist_loss + dim_loss + obj_loss + class_loss

# Extract i:th cell's objectness scores
def extract_cell_objectness_scores(tensor, i):
  objectness_scores = []
  for j in range(B):
    cell_x = i % S
    cell_y = math.floor(i / S)
    objectness_scores.append(tensor[cell_x, cell_y, 5 * j])   # index is "0 + 5*j"
  return objectness_scores

# Extract i:th cell's bounding boxes (in YOLO form) from tensor
def extract_cell_bounding_boxes(tensor, i):
  bounding_boxes = []
  for j in range(B):
    cell_x = i % S
    cell_y = math.floor(i / S)
    x = tensor[cell_x, cell_y, 1 + 5 * j]
    y = tensor[cell_x, cell_y, 2 + 5 * j]
    sqrtw = tensor[cell_x, cell_y, 3 + 5 * j]
    sqrth = tensor[cell_x, cell_y, 4 + 5 * j]
    bounding_boxes.append((x, y, sqrtw, sqrth, i))
  return bounding_boxes

# Extract i:th cell's class probabilities from tensor
def extract_cell_class_probabilities(tensor, i):
  cell_x = i % S
  cell_y = math.floor(i / S)
  return tensor[cell_x, cell_y, B*5:]

# Convert box in "label" form to YOLO form
# TODO: Verify functionality
def to_YOLO_box(xmin, ymin, xmax, ymax):
  xcenter = (xmin + xmax) / 2
  ycenter = (ymin + ymax) / 2
  x = (xcenter % cell_w) / cell_w # equiv. to '(xcenter / cell_w) % 1', I think
  y = (ycenter % cell_h) / cell_h
  w = xmax - xmin
  h = ymax - ymin
  i = math.floor(xcenter/cell_w) + S*math.floor(ycenter/cell_h)
  return (x, y, math.sqrt(w), math.sqrt(h), i)

# Convert box in YOLO form to "label" form
# i: cell index
# TODO: Verify functionality
def from_YOLO_box(x, y, sqrtw, sqrth, i):
  cell_x = i % S
  cell_y = math.floor(i / S)
  xmin = cell_x * cell_w + x * cell_w - (sqrtw**2)/2
  xmax = cell_x * cell_w + x * cell_w + (sqrtw**2)/2
  ymin = cell_y * cell_h + y * cell_h - (sqrth**2)/2
  ymax = cell_y * cell_h + y * cell_h + (sqrth**2)/2
  return (xmin, ymin, xmax, ymax)

# Intersection over union
# Algorithm from: https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c
# TODO: Verify functionality
def IoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
  intersect_w = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
  intersect_h = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
  intersect = intersect_h * intersect_w
  union = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - intersect
  return intersect / union

def extract_target_from_file(label_file):
  
  dom = minidom.parse(label_file)
  
  # Compute factor to scale dimensions by
  actual_size = dom.getElementsByTagName('size')[0]
  actual_width = float(actual_size.getElementsByTagName('width')[0].firstChild.nodeValue)
  actual_height = float(actual_size.getElementsByTagName('height')[0].firstChild.nodeValue)
  width_scale = img_width/actual_width
  height_scale = img_height/actual_height

  # Extract box position and label from file  
  objects = dom.getElementsByTagName('object')
  boxes = []
  class_probs = []
  for object in objects:
    xmin  = float(object.getElementsByTagName('xmin')[0].firstChild.nodeValue) * width_scale
    ymin  = float(object.getElementsByTagName('ymin')[0].firstChild.nodeValue) * height_scale
    xmax  = float(object.getElementsByTagName('xmax')[0].firstChild.nodeValue) * width_scale
    ymax  = float(object.getElementsByTagName('ymax')[0].firstChild.nodeValue) * height_scale
    label = object.getElementsByTagName('name')[0].firstChild.nodeValue
    class_prob = np.zeros(len(classes))
    for i in range(len(classes)):
      if label == classes[i]:
        class_prob[i] = 1
        break
    boxes.append(to_YOLO_box(xmin, ymin, xmax, ymax))
    class_probs.append(class_prob)

  target = np.zeros((S, S, (B*5 + C)))

  # Form target tensor
  for (box, class_prob) in zip(boxes, class_probs):
    i = box[4]
    cell_x = i % S
    cell_y = math.floor(i/S)
    target[cell_x, cell_y, 0] = 1
    target[cell_x, cell_y, 1] = box[0]
    target[cell_x, cell_y, 2] = box[1]
    target[cell_x, cell_y, 3] = box[2]
    target[cell_x, cell_y, 4] = box[3]

    for (i, prob) in enumerate(class_prob):
      target[cell_x, cell_y, B*5 + i] = prob
    
  return target

# Define model
# NOTE:
# Interpreting blocks as input/output of layers. Mini-blocks represent filters/kernels (conv),
# and crossing arrows represent fully-connected layers. (Confirmed)
# NOTE: 
# Interpreting '2x2-s-2' as '2x2 stride: 2' (Confirmed)
# NOTE:
# Only setting stride of conv layers to 2x2 where explicit in paper. 
# Using default of 1x1 everywhere else. (Confirmed)
class YOLOv1(tf.keras.Model):
  def __init__(self):
    super(YOLOv1, self).__init__()

    self.net = [
      tf.keras.layers.Resizing(img_height, img_width),
      tf.keras.layers.Rescaling(1./255),

      # Block 1
      tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 2
      tf.keras.layers.Conv2D(filters=192, kernel_size=3, padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 3
      tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 4
      tf.keras.layers.Conv2D(filters=256,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=256,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=256,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=256,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 5
      tf.keras.layers.Conv2D(filters=512,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=512,  kernel_size=1, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 6
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Block 7
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(4096),
      tf.keras.layers.LeakyReLU(alpha=0.1),

      # Output
      tf.keras.layers.Dense(S*S*(B*5 + C)), # Linear activation <=> no activation at all
      tf.keras.layers.Reshape((S, S, B*5 + C))
    ]

  def call(self, inputs):
    x = inputs
    for (i, layer) in enumerate(self.net):
      x = layer(x)
    return x

model = YOLOv1()
print("Loading checkpoint")
model.load_weights(checkpoint_path)

# ADAM optimizer
optimizer = tf.keras.optimizers.Adam(
  learning_rate=LR,
  beta_1=0.9,
  beta_2=0.999,
)

# From: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/
for epoch in range(EPOCHS):
  print("-- Epoch " + str(epoch+1) + " --")

  # Batch of inputs and targets
  x_train_samples = []
  y_train_samples = []

  step = 0

  # Fetch batch
  for label_dir in label_dirs:
    for file in os.listdir(label_dir):
      if not isfile(join(label_dir, file)):
        print("Not a file! Skipping. (" + file + ")")
        continue

      step += 1
      
      # Save checkpoint
      if step % SAVE_INTERVAL == 0:
        print("Saving checkpoint")
        model.save_weights(checkpoint_path)

      # Add formatted target (from label file)
      label_file = label_dir + "/" + file
      target = extract_target_from_file(label_file)
      y_train_samples.append(target)

      # Remove last folder ('labels') from label_dir
      img_dir, _ = os.path.split(label_dir)

      # Add input image
      img = tf.keras.utils.load_img(img_dir + "/" + os.path.splitext(file)[0] + ".jpeg")
      x_train_samples.append(tf.keras.utils.img_to_array(img))
    
      # Train on batch
      if len(x_train_samples) >= BATCH_SIZE:

        print(str(len(x_train_samples)) + ": " + str(x_train_samples[0].shape))
        print(str(len(y_train_samples)) + ": " + str(y_train_samples[0].shape))

        # Convert list of 3D tensors into 4D tensor
        x_train = np.stack(x_train_samples)
        y_train = np.stack(y_train_samples)

        print("x_train shape: " + str(x_train.shape))
        print("y_train shape: " + str(y_train.shape))

        with tf.GradientTape() as tape:
          out = model(x_train, training=True)
          loss_val = YOLOv1_loss(y_train, out)
          print("Loss: " + str(loss_val))

        grads = tape.gradient(loss_val, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        x_train_samples.clear()
        y_train_samples.clear()
