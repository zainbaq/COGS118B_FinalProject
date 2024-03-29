import numpy as np 
import pandas as pd 

################################################################
def process_names(st):
    st = list(st)
    for i in range(len(st)):
        if st[-1] == '.':
            break
        else:
            st.pop()
    st.pop()
    st = ''.join(st)
    st = st.replace('.', '-')
    return st

def train_test_split(df):
    
    df.sample(frac=1)
    print('Data loaded and shuffled.')

    # n_neighbors = int(input("n_neighbors: "))
    n_obs = df.shape[0]
    train_frac = 0.7

    trainset = df.iloc[:int(train_frac*n_obs), :]
    testset = df.iloc[int(train_frac*n_obs):n_obs-1, :]

    train_labels = trainset.expression
    train_inputs = trainset.drop(columns=['expression', 'Unnamed: 0'], axis=1)

    test_labels = testset.expression
    test_inputs = testset.drop(columns=['expression', 'Unnamed: 0'], axis=1)

    print("\nTrain inputs shape: {0}, train labels shape: {1}".format(
        train_inputs.shape,
        train_labels.shape
    ))

    print("\nTest inputs shape: {0}, test labels shape: {1}".format(
        test_inputs.shape,
        test_labels.shape
    ))

    return train_inputs, train_labels, test_inputs, test_labels

# Loading in saved data
data_dir = 'datasets/jaffe_loaded.npy'
names_dir = 'datasets/jaffe_img_ids.txt'
labels_dir = 'datasets/jaffe_labels.csv'

df = pd.DataFrame(data=np.load(data_dir))
names = pd.read_csv(names_dir, delimiter=',', header=None).to_numpy().squeeze()

# Manipulate image names to match label format
names = [process_names(name) for name in names]

# Add names column to image df
df['names'] = names

df_labels = pd.read_csv(labels_dir)
df_labels = df_labels.drop(columns='Unnamed: 0', axis=1)

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

df_labels['names'] = df_labelids
df_labels['expression'] = expressions

df_all = pd.merge(df, df_labels, on='names')

df_final = df_all.drop(
    columns=['names', 'HAP', 'SAD', 'SUR', 'ANG', 'DIS', 'FEA'],
    axis=1
)

df_final.to_csv(
    'datasets/final_data.csv',
    sep=','
)
#################################################################

