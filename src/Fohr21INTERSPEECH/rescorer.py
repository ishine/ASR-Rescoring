import sys
import torch
from tqdm import tqdm
from jiwer import cer

from util.dataparser import DataParser

class Rescorer():
    def __init__(self, config) -> None:
        print("Setting rescorer ...")
        self.config = config

    def parse_data(self, data_path):
        data_parser = DataParser(json_file_path=data_path)
        data = data_parser.parse()
        return data

    def find_best_weight(self, p_ranges: tuple, scores: tuple, step=1):

        best_cer = sys.float_info.max
        for p_range, score in zip(p_ranges, scores):


        for alpha in tqdm(range(p_ranges[0][0], p_ranges[0][1]+1, step)):
            for beta in tqdm(range(p_ranges[1][0], p_ranges[1][1]+1, step), leave=False):
                for gamma in tqdm(range(p_ranges[2][0], p_ranges[2][1]+1, step), leave=False):
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
        print("Parsing dev and test data ...")
        self.scores = {}
        for score_type in self.config.use_score:
            input_file = getattr(self.config.input_files, score_type)
            self.scores[score_type] = {}
            for file_type in ["dev", "test"]:
                path = getattr(input_file, file_type)
                data = self.parse_data(path)
                self.scores[score_type][file_type] = torch.tensor(data.get_scores())

        self.p_range = {}
        for score_type in self.config.use_score:
            range = getattr(self.config.parameter_range, score_type)
            self.p_range[score_type] = {}
            start = getattr(range, "start")
            end = getattr(range, "end")
            self.p_range[score_type] = (start, end)

        print("Using dev to find best weight ...")
        best_weights = self.find_best_weight(p_range=())

        print("Computing test score ...")
        combined_score = self.combine_score()
        hyps_ids = combined_score.argmax(dim=1)
        
        ref_texts, hyp_texts = [], []
        for utt, hyp_idx in zip(self.dev_am_data.utt_set, hyps_ids):
            ref_texts.append(utt.ref.text)
            hyp_texts.append(utt.hyps[hyp_idx].text)
        
        cer = cer(ref_texts, hyp_texts)
        print("test cer: ", cer)
