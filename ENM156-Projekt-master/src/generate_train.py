import os
import random

train_files = []
test_files = []

test_split = 0.2

sample_count = 0

# Find all files
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
  if filename.endswith(".jpeg") or filename.endswith(".jpg"):
    train_files.append("../data/obj/" + filename)
    sample_count += 1
os.chdir("..")

# Use some portion of files for testing
for i in range(int(sample_count*test_split)):
  rand_idx = random.randrange(len(train_files))
  test_files.append(train_files[rand_idx])
  train_files.pop(rand_idx)

# Generate train file
with open("train.txt", "w") as outfile:
  for image in train_files:
      outfile.write(image)
      outfile.write("\n")
  outfile.close()

# Generate test file
with open("test.txt", "w") as outfile:
  for image in test_files:
      outfile.write(image)
      outfile.write("\n")
  outfile.close()

tot = len(train_files) + len(test_files)
print("Before augmentation:")
print("Training on " + str(len(train_files)) + " of " + str(tot) + " images")
print("Testing on " + str(len(test_files)) + " of " + str(tot) + " images")

os.chdir("..")