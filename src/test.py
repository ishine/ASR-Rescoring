from ssl import OP_NO_SSLv2
import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from models import sentence_bert_lm
from train import load_model
from sklearn.preprocessing import normalize
from util.minimum_edit_distance import minimum_edit_distance

ref = list("abcde")
hyp = list("cdef")


output = minimum_edit_distance(hyp, ref)


ref = []
hyp = []
ops = []
for (ref_token, hyp_token, op_token) in output:
    ref.append(ref_token)
    hyp.append(hyp_token)
    ops.append(op_token)
print(ref)
print(hyp)
print(ops)