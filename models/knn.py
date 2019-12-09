from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pandas as pd 
from train_test_split import train_test_split
import os

data_dir = 'datasets/final_data.csv'

df = pd.read_csv(data_dir, sep=',')

tr_inp, tr_labels, te_inp, te_labels = train_test_split(df)

n_neighbors = range(1, tr_inp.shape[0])
# n_neighbors = int(input("n_neighbors: "))

acc_list = np.array([])
for n in n_neighbors:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(tr_inp, tr_labels)

    acc = clf.score(te_inp, te_labels)
    print("\nKNN = {0} accuracy: {1}".format(n, acc))

    acc_list = np.append(acc_list, acc)

print("\nBest K: {0}, with accuracy {1}".format(
    n_neighbors[np.argmax(acc_list)],
    np.max(acc_list)
))



