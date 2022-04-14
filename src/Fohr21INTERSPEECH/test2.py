import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9,10,11])

step = 4
l = len(a)%step
print(len(a)%step)
print(a[:-1*(len(a)%step)])
