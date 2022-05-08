import torch

pos = torch.tensor([[1,3,5], [2,4,6]])
output = torch.tensor([[1,2,3,4,5,6,0,0], [4,5,6,8,9,10,11,12]])

for idx, p in enumerate(pos):
    print(output[idx, p])

