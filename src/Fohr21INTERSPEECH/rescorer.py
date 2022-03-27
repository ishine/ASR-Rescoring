import sys
import torch
from tqdm import tqdm
from jiwer import cer

from util.dataparser import DataParser

class Rescorer():
    def __init__(self, config) -> None:
        print("Setting rescorer ...")
        self.config = config

        if len(self.config.use_score) == 0:
            print("number of score type is invalid, should be 1~3")
            return

        
        '''        
        if  hasattr(self.config, "lm_data_file"):
            self.lm_data_file = self.config.lm_data_file
        else:
            self.lm_data_file = None
            print("without lm_data_file")

        if  hasattr(self.config, "sem_data_file"):
            self.sem_data_file = self.config.sem_data_file
        else:
            self.sem_data_file = None
            print("without sem_data_file")
        '''

    def parse_data(self, data_path):
        data_parser = DataParser(json_file_path=data_path)
        data = data_parser.parse()
        return data

    def find_best_weight(
        self,
        alpha_range=(1,1),
        beta_range=(8,10),
        gamma_range=(0,100),
        step=1,
        *scores
    ):
        best_cer = sys.float_info.max
        for alpha in tqdm(range(alpha_range[0], alpha_range[1]+1, step)):
            for beta in tqdm(range(beta_range[0], beta_range[1]+1, step), leave=False):
                for gamma in tqdm(range(gamma_range[0], gamma_range[1]+1, step), leave=False):
                    combined_score = self.combine_score((alpha, beta, gamma), scores)
                    hyps_ids = combined_score.argmax(dim=1)
                    
                    ref_texts, hyp_texts = [], []
                    for utt, hyp_idx in zip(self.dev_am_data.utt_set, hyps_ids):
                        ref_texts.append(utt.ref.text)
                        hyp_texts.append(utt.hyps[hyp_idx].text)
                    
                    cer = cer(ref_texts, hyp_texts)
                    if cer < best_cer:
                        best_cer = cer
                        best_weights = (alpha, beta, gamma)
        return best_weights

    def combine_score(self, weights, scores):
        combined_score = 1
        for weight, score in zip(weights, scores):
            combined_score *= torch.pow(score, weight)
        return combined_score

    def rescore(self):
        print("Parsing dev data ...")
        self.dev_am_data = self.parse_data(self.config.dev.am_data_file)
        self.dev_lm_data = self.parse_data(self.config.dev.lm_data_file)
        self.dev_sem_data = self.parse_data(self.config.dev.sem_data_file)

        print("Parsing test data ...")
        self.test_am_data = self.parse_data(self.config.test.am_data_file)
        self.test_lm_data = self.parse_data(self.config.test.lm_data_file)
        self.test_sem_data = self.parse_data(self.config.test.sem_data_file)

        print("Using dev to find best weight ...")
        dev_am_score = torch.tensor(self.dev_am_data.get_scores())
        dev_lm_data = torch.tensor(self.dev_lm_data.get_scores())
        dev_sem_score = torch.tensor(self.dev_sem_data.get_scores())
        best_weights = self.find_best_weight(scores=(dev_am_score, dev_lm_data, dev_sem_score))

        print("")
        test_am_score = torch.tensor(self.test_am_data.get_scores())
        test_lm_score = torch.tensor(self.test_lm_data.get_scores())
        test_sem_score = torch.tensor(self.test_sem_data.get_scores())
        combined_score = self.combine_score(best_weights, (test_am_score, test_lm_score, test_sem_score))
        hyps_ids = combined_score.argmax(dim=1)
        
        ref_texts, hyp_texts = [], []
        for utt, hyp_idx in zip(self.dev_am_data.utt_set, hyps_ids):
            ref_texts.append(utt.ref.text)
            hyp_texts.append(utt.hyps[hyp_idx].text)
        
        cer = cer(ref_texts, hyp_texts)
        print("test cer: ", cer)
