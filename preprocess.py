import numpy as np 
import pandas as pd
from PIL import Image

import os
import csv 

jaffe_dir = 'datasets/raw/jaffedbase/'

# Load in data
data = np.array([])
img_names = np.array([])
d_size = 0
for filename in os.listdir(jaffe_dir):
    if filename.endswith(".tiff"):
        im = Image.open(os.path.join(jaffe_dir, filename))
        imarray = np.array(im)
        data = np.append(data, imarray)
        name = filename.replace('.tiff', '')
        img_names = np.append(img_names, name)
        d_size = d_size+1

# save img names to separate csv
with open('datasets/jaffe_img_ids.txt', 'w') as F:
    wr = csv.writer(F)
    wr.writerow(img_names)

# reshape data such that each row is a 256x256 image of a face
data = np.reshape(data, (d_size, -1))
# Save data to avoid future reading
np.save('datasets/jaffe_loaded.npy', data)
np.savetxt('datasets/jaffe_img_ids.csv', data, delimiter=',')





