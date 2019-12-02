import numpy as np 
import pandas as pd 

def process_names(st):
    st = list(st)
    for i in range(len(st)):
        if st[-1] == '.':
            break
        else:
            st.pop()
    st.pop()
    st = ''.join(st)
    return st

data_dir = 'datasets/jaffe_loaded.npy'
names_dir = 'datasets/jaffe_img_ids.txt'
labels_dir = 'datasets/jaffe_labels.csv'

df = pd.DataFrame(data=np.load(data_dir))
names = pd.read_csv(names_dir, delimiter=',', header=None).to_numpy().squeeze()

names = [process_names(name) for name in names]

df['names'] = names

df_labels = pd.read_csv(labels_dir)
df_labels = df_labels.drop(columns='Unnamed: 0', axis=1)

print(df.head(5))
print(df_labels)

'''
The labels are given as a table of 6 mean scores for each image.
Therefore, we take the expression with the highest mean score among these
six, and use that as the label for that image.
'''

df_labelids = df_labels.PIC
df_labels = df_labels.drop(columns=['PIC', '#'], axis=1)

expressions = []
for i, row in enumerate(df_labels.to_numpy()):
    # print(type(row))
    expression = np.argmax(row)
    expressions.append(expression)

df_labels['PIC'] = df_labelids
df_labels['expression'] = expressions
print(df_labels.expression.value_counts())

print(df_labels)
