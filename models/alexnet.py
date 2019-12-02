import torch
from PIL import Image
from torchvision import transforms

from train_test_split import train_test_split
import pandas as pd 
import numpy as np 

data_dir = 'datasets/final_data.csv'

df = pd.read_csv(data_dir, sep=',')

tr_inp, tr_labels, te_inp, te_labels = train_test_split(df)

model = torch.hub.load(
    'pytorch/vision:v0.4.2', 
    'alexnet',
    pretrained=True
)

model.eval()

in_img = Image.open(tr_inp.iloc[0])