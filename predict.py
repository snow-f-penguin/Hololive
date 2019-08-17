import sys, os
from PIL import Image
from keras import models
import numpy as np

if len(sys.argv) <= 1:
  quit()

image_size = 64 
input_dir = 'data/train'
charactors = ['tokinosora','fubuki','roboco','matsuri','aki','choco','meru','aqua','subaru','ayame','shion','ha-to','mio','miko']

X = []
for file_name in sys.argv[1:]:
  img = Image.open(file_name)
  img = img.convert("RGB")
  img = img.resize((image_size, image_size))
  in_data = np.asarray(img)
  X.append(in_data)

X = np.array(X)

model = models.load_model('./model/hololive.hdf5', compile=False)
predict = model.predict(X)

for pre in predict:
  y = pre.argmax()
  print("Hololiver Name : ", charactors[y])
