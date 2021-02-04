import pandas as pd
import os
par_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(par_dir)
a = pd.read_csv('PeMS-M/W_170.csv',header=None).values

print(a[0,1])
