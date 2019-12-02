import numpy as np 
import pandas as pd
from PIL import Image

import os

jaffe_dir = 'datasets/jaffe/jaffedbase/'

# Load in data
data = np.array([])
d_size = 0
for filename in os.listdir(jaffe_dir):
    if filename.endswith(".tiff"):
        im = Image.open(os.path.join(jaffe_dir, filename))
        imarray = np.array(im)
        data = np.append(data, imarray)
        d_size = d_size+1

# reshape data such that each row is a 256x256 image of a face
data = np.reshape(data, (d_size, -1))
# Save data to avoid future reading
np.save('datasets/jaffe_loaded.npy', data)

label_dir = 'datasets/jaffe/labels.txt'

labels = pd.read_csv(label_dir, sep=" ", header=None)
labels.columns = ['#', 'HAP', 'SAD', 'SUR', 'ANG', 'DIS', 'FEA', 'PIC']

labels.to_csv('datasets/jaffe_labels.csv', ',')




