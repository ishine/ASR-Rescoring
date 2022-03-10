import sys
import numpy as np
from typing import List
from jiwer import cer
from tqdm import tqdm
from util.parse_json import parse_json
from sklearn.preprocessing import normalize
def length_penalty(alpha, sentence: str = None, sentences: list = None):
    
    if isinstance(sentence, str):
        seq_len = len(sentence)
        LP = ((5+seq_len)^alpha) / ((5+1)^alpha)
    
    elif isinstance(sentences, list):
        LP = []
        for sentence in sentences:
            seq_len = len(sentence)
            LP.append( ((5+seq_len)^alpha) / ((5+1)^alpha) )

    return LP

class Rescorer():
    def __init__(self, config):
        self.config = config

    def find_best_weight(self):
        dev_ASR_score, dev_ref_text, dev_hyp_text = parse_json(
            self.config.dev_asr_data_path,
            requirements=["hyp_score", "ref_text", "hyp_text"],
            max_utts=self.config.max_utts)

        dev_LM_score = parse_json(
            self.config.dev_lm_data_path,
            requirements=["hyp_score"],
            max_utts=self.config.max_utts)

        best_cer = sys.float_info.max

        for weight in tqdm(np.arange(0.0, 1.0, 0.01)):
            # 將ASR分數和LM分數做 weighted sum(rescore)
            final_score = self.rescore(weight, dev_ASR_score, dev_LM_score, dev_hyp_text)
           
            # 取出最高分的hyp sentences
            predict_text = self.get_highest_score_hyp(final_score, dev_hyp_text)

            # 計算error rate
            error = cer(dev_ref_text, predict_text)

            if error < best_cer:
                best_cer = error
                best_weight = weight

        return best_weight, best_cer

    def rescore(self, weight, ASR_score, LM_score, hyp_text):
        LM_score = np.array(LM_score)
        ASR_score = np.array(ASR_score)

        hyp_text = np.array(hyp_text)
        hyp_test_shape = hyp_text.shape
        hyp_text = hyp_text.reshape(-1)

        LP = length_penalty(alpha=1, sentences=list(hyp_text))
        LP = np.array(LP).reshape(hyp_test_shape)

        final_score = (1-weight)*ASR_score + weight*LM_score/LP
        #ASR_score = normalize(ASR_score, norm = "max", axis=1)
        #LM_score = normalize(LM_score, norm = "max", axis=1)
        #final_score = (1-weight)*ASR_score + weight*LM_score
        return final_score
    
    def get_highest_score_hyp(self, final_score, dev_hyp_text):
        
        max_score_hyp_index = np.argmax(final_score, axis=-1)
        best_hyp = [ht[index] for ht, index in zip(dev_hyp_text, max_score_hyp_index)]
        return best_hyp