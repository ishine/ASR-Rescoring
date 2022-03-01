import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from models import sentence_bert_lm
from train import load_model
from sklearn.preprocessing import normalize
'''
model = load_model("bert-base-chinese", "/home/chkuo/chkuo/experiment/rescoring/result/checkpoint_10.pth")
model(input_ids = torch.tensor([[[1,2,3],[11,2,3],[21,2,3]]]))
'''
a = [[1,2],[3,4]]
a = np.array(a)
a = normalize(a, norm="max", axis=0)
print(a)