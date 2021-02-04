import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs

a = np.array([[2, 3, 4],
              [3, 5, 6]], dtype=np.float)
c = np.array([[2, 2, 1]])

# b = eigs(a, k=1, which='LR')[0][0].real
b = eigs(a, k=1, which='LR')
print(np.concatenate)
