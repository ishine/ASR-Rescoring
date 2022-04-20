import sys
import itertools
import torch
from tqdm import tqdm
from jiwer import cer

from util.dataparser import DataParser
from util.plot import plot
class Rescorer():
    def __init__(self, config) -> None:
        print("Setting rescorer ...")
        self.config = config

    def parse_data(self, data_path, max_utts, n_best):
        data_parser = DataParser(data_path, max_utts, n_best)
        data = data_parser.parse()
        return data

    def find_best_weight(self, text_data, weight_pairs:list, scores: list):
        cer_history = []
        best_cer = sys.float_info.max
        for weight_pair in tqdm(weight_pairs, total=len(weight_pairs)):
            final_score = self.combine_score(weight_pair, scores)
            hyps_ids = final_score.argmax(dim=1)
            
            ref_texts, hyp_texts = [], []
            #for utt, hyp_idx in zip(self.dev_am_data.utt_set, hyps_ids):
            for utt, hyp_idx in zip(text_data, hyps_ids):
                ref_texts.append(utt.ref.text)
                hyp_texts.append(utt.hyps[hyp_idx].text)
            
            error_rate = cer(ref_texts, hyp_texts)
            cer_history.append(error_rate)
            if error_rate < best_cer:
                best_cer = error_rate
                best_weights = weight_pair

        return best_weights, best_cer, cer_history

    def combine_score(self, weight_pair, scores):
        final_score = 1
        for weight, score in zip(weight_pair, scores):
            final_score *= torch.pow(score, weight)
        return final_score

    def rescore(self):
        print("Parsing dev and test data ...")
        self.scores = {}
        for file_type in ["dev", "test"]:
            self.scores[file_type] = []
            ft = getattr(self.config.input_files, file_type)
            for score_type in self.config.use_score:
                path = getattr(ft, score_type)
                data = self.parse_data(path, self.config.max_utts, self.config.n_best)

                score = torch.tensor(data.get_scores(), dtype=torch.double)
                if score_type == "am":
                    setattr(self, f"{file_type}_{score_type}_data", data)
                    # shift score，因為本am score為負數
                    score -= (torch.min(score, dim=1, keepdim=True).values - 2)

                self.scores[file_type].append(score)

        self.weight_range = []
        for score_type in self.config.use_score:
            r = getattr(self.config.weight_range, score_type)
            start = getattr(r, "start")
            end = getattr(r, "end")
            self.weight_range.append(list(range(start, end+1)))

        self.weight_pairs = list(itertools.product(*self.weight_range))
            
        print("Using dev to find best weight ...")
        best_weights, best_cer, dev_cer_history = self.find_best_weight(
            self.dev_am_data.utt_set,
            self.weight_pairs,
            self.scores["dev"]
        )
        
        print("best weights: ", best_weights)
        print("best dev cer: ", best_cer)
        
        print("Computing test score ...")
        combined_score = self.combine_score(best_weights, self.scores["test"])
        hyps_ids = combined_score.argmax(dim=1)
        
        ref_texts, hyp_texts = [], []
        for utt, hyp_idx in zip(self.test_am_data.utt_set, hyps_ids):
            ref_texts.append(utt.ref.text)
            hyp_texts.append(utt.hyps[hyp_idx].text)
        
        error_rate = cer(ref_texts, hyp_texts)
        print("test cer: ", error_rate)



        test_best_weights, test_best_cer, test_cer_history = self.find_best_weight(
            self.test_am_data.utt_set,
            self.weight_pairs,
            self.scores["test"]
        )
        print("best (cheating) test set weights: ", test_best_weights)
        print("best (cheating) test set score: ", test_best_cer)
        plot(
            self.config.output_file.cer_weight_curve_file,
            ["dev", "test"], 
            [dev_cer_history, test_cer_history],
            {"x": "weight", "y": "CER"}
        )