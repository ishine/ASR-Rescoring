import numpy as np
from sklearn.preprocessing import normalize
import math
v_list = [0,1,2,3,4,5,6,7,8,9]
'''
normalized_v, _ = normalize(v[:,np.newaxis], norm = "l2", axis=0, return_norm=True)
print(normalized_v.ravel())
print(_)
'''
normalize_v = [math.log(v+1) for v in v_list]
print(normalize_v)