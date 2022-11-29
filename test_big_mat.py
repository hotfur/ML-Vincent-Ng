# Test ultra big matrices multiplication
import numpy as np

dim = 30000
big_mat = np.empty((dim, dim), dtype=np.float32)
big_mat2 = np.empty((dim, dim), dtype=np.float32)
result = big_mat@big_mat2
