import pandas as pd

def train_test_split(df):
    
    df.sample(frac=1)
    print('Data loaded and shuffled.')

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