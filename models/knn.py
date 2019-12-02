from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pandas as pd 
from train_test_split import train_test_split
import os

data_dir = 'datasets/final_data.csv'

df = pd.read_csv(data_dir, sep=',')

tr_inp, tr_labels, te_inp, te_labels = train_test_split(df)

n_neighbors = int(input("n_neighbors: "))

clf = KNeighborsClassifier(n_neighbors=n_neighbors)
