import sys

from numpy import pad
sys.path.append("/home/chkuo/chkuo/experiment/ASR-Rescoring/src")
from models.semantic_bert import SemanticBert 

import torch
from models.semantic_bert import AMLMSemanticBert 
input = torch.tensor([
    [[ 1,  2,  3],
     [ 4,  5,  6],
     [ 7,  8,  9]], 
    
    [[11, 12, 13],
     [14, 15, 16],
     [17, 18, 19]]
])

# pad_mask size => (batch size, sequence length)
pad_mask = torch.tensor([[1,1,0],[1,1,1]])

# 
pad_mask = torch.unsqueeze(pad_mask, -1).expand(input.size())
print(torch.masked_select(input, pad_mask.ge(0.5)))
#model = SemanticBert(bert_model="bert-base-chinese")

'''
def avg_pooling(input_tensor, pad_mask):

'''