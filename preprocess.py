import numpy as np 
import pandas as pd
from PIL import Image

import os
import csv 

def load_data(data_dir, labels_dir, labels_target_dir):
    # Load in data
    data = np.array([])
    img_names = np.array([])
    d_size = 0
    for filename in os.listdir(jaffe_dir):
        if filename.endswith(".tiff"):
            im = Image.open(os.path.join(jaffe_dir, filename))
            # Convert tiff to rgb
            im_rgb = Image.new("RGB", im.size)
            im_rgb.paste(im)
            imarray = np.array(im_rgb)
            data = np.append(data, imarray)
            name = filename.replace('.tiff', '')
            img_names = np.append(img_names, name)
            d_size = d_size+1

    label_head = ['#', 'HAP', 'SAD', 'SUR', 'ANG', 'DIS', 'FEA', 'PIC']
    labels = pd.read_csv(
        labels_dir, 
        sep=' ', 
        header=None,
        names = label_head
        )
    # reshape data such that each row is a 256x256 image of a face
    data = np.reshape(data, (d_size, -1))

    return data, labels, img_names

def save_data(data, labels, img_names):
    # Save data to avoid future reading
    np.save('datasets/jaffe_loaded.npy', data)
    # Save labels to separate csv
    labels.to_csv(labels_target_dir, sep=',')
    # save img names to separate csv
    with open('datasets/jaffe_img_ids.txt', 'w') as F:
        wr = csv.writer(F)
        wr.writerow(img_names)

if __name__ == '__main__':

    jaffe_dir = 'datasets/jaffe/jaffedbase/'
    labels_dir = 'datasets/jaffe/labels.txt'
    labels_target_dir = 'datasets/jaffe_labels.csv'

    data, labels, img_names = load_data(
        jaffe_dir,
        labels_dir,
        labels_target_dir
    )

    save_data(data, labels, img_names)



