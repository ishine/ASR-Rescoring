import numpy as np
a = np.array([2,1,0])
print("a: ", a)
print("softmax result: ",np.exp(-1*a) / np.sum(np.exp(-1*a)))

b = a - np.average(a)
print("b: ", b)
print("softmax result: ", np.exp(-1*b) / np.sum(np.exp(-1*b)))

