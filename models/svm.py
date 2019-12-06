import numpy as np 
import pandas as pd 

from train_test_split import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

'''This file defines a Support Vector Machine to be
used with this dataset'''

# Methods

def train_SVM(X, y, kernel):
    clf = SVC(kernel=kernel)
    clf.fit(X, y)
    return clf

# Training

kernel = 'linear'
data_dir = 'datasets/final_data.csv'

df = pd.read_csv(data_dir, sep=',')

tr_inp, tr_labels, te_inp, te_labels = train_test_split(df)

clf = train_SVM(tr_inp, tr_labels, kernel)

y_hat_tr = clf.predict(tr_inp)
y_hat_te = clf.predict(te_inp)

tr_report = classification_report(tr_labels, y_hat_tr)
te_report = classification_report(te_labels, y_hat_te)

print("\n Training Report: \n{0}, \n Test Report: \n{1}".format(
    tr_report,
    te_report
))

