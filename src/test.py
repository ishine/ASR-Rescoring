from ssl import OP_NO_SSLv2
import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from models import sentence_bert_lm
from train import load_model
from sklearn.preprocessing import normalize
from util.minimum_edit_distance import minimum_edit_distance
a = torch.tensor([])
a = torch.cat((a, torch.tensor([1,2]))) 
print( torch.cat((a, torch.tensor([1,2]))) )
print(max(a.to(torch.int)))
print([0]*max(a.to(torch.int)))