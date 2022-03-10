import torch
from preprocessing.levenshtein import levenshtein_distance_alignment
from util.parse_json import parse_json

def compute_loss(asr_scores, LM_scores, hyp_cer):
    #print(asr_scores)
    #print(LM_scores)
    #print(hyp_cer)
    final_scores = asr_scores + LM_scores
    #print(final_scores)
    
    error_distribution = torch.softmax(hyp_cer, dim=1)
    print(error_distribution)
    
    #print(torch.sum(final_scores,dim=1).unsqueeze(dim=1))
    #print(torch.sum(hyp_cer,dim=1).unsqueeze(dim=1))
    temperature = torch.sum(final_scores,dim=1) / torch.sum(hyp_cer,dim=1)
    temperature = temperature.unsqueeze(dim=1)
    #print(temperature)
    
    #print(final_scores/temperature)
    score_distribution = torch.softmax(final_scores/temperature, dim=1)
    print(score_distribution)
    
    batch_loss = torch.kl_div(score_distribution, error_distribution, reduction="sum")
    print(batch_loss)
    
    return 0

asr_scores = torch.tensor([[1,2,3], [4,5,6]])
LM_scores = torch.tensor([[3,4,5], [6,7,8]])
hyp_cer = torch.tensor([[0.5, 0.4, 0.7], [0.6, 0.5, 0.8]])
compute_loss(asr_scores, LM_scores, hyp_cer)