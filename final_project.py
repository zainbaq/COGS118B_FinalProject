import numpy as np 
import pandas as pd 

data_dir = 'datasets/clean/jaffe_loaded.npy'

df = pd.DataFrame(data=np.load(data_dir))

print(df.head(10))

