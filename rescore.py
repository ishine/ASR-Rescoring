import sys
import json

import numpy as np
from jiwer import cer 
from tqdm import tqdm

from util.arg_parser import ArgParser

def dict_to_list(dict):
    all_scores = []
    for utt_id, hyps in dict.items():
        if isinstance(hyps, dict):
            row_scores = []
            for hyp_id, hyp in hyps.items():
                row_scores.append(hyp)
            all_scores.append(row_scores)
        else:
            all_scores.append(hyps)
    return all_scores

def find_best_weight(am, lm, hyps, ref):
    best_cer = sys.float_info.max

    for weight in tqdm(np.arange(0.0, 1.0, 0.01)):
        final_score = rescore(weight, am, lm)
        predict_hyps = get_highest_score_hyp(final_score, hyps)
        error = cer(ref, predict_hyps)

        if error < best_cer:
            best_cer = error
            best_weight = weight

    return best_weight, best_cer

def rescore(weight, ASR_score, LM_score):
    LM_score = np.array(LM_score)
    ASR_score = np.array(ASR_score)
    final_score = (1-weight)*ASR_score + weight*LM_score
    return final_score

def get_highest_score_hyp(final_score, hyps):
    max_score_hyp_index = np.argmax(final_score, axis=-1)
    best_hyp = [ht[index] for ht, index in zip(hyps, max_score_hyp_index)]
    return best_hyp


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    dev_am = dict_to_list(json.load(
        open(config.dev_am_path, "r", encoding="utf-8")
    ))
    dev_lm = dict_to_list(json.load(
        open(config.dev_lm_path, "r", encoding="utf-8")
    ))
    dev_hyps = dict_to_list(json.load(
        open(config.dev_hyps_text_path, "r", encoding="utf-8")
    ))
    dev_ref = dict_to_list(json.load(
        open(config.dev_ref_text_path, "r", encoding="utf-8")
    ))
    best_weight, best_cer = find_best_weight(dev_am, dev_lm, dev_hyps, dev_ref)
    print("best_weight: ", best_weight)
    print("dev cer: ", best_cer)

    test_am = dict_to_list(json.load(
        open(config.dev_am_path, "r", encoding="utf-8")
    ))
    test_lm = dict_to_list(json.load(
        open(config.dev_lm_path, "r", encoding="utf-8")
    ))
    test_hyps = dict_to_list(json.load(
        open(config.dev_hyps_text_path, "r", encoding="utf-8")
    ))
    test_ref = dict_to_list(json.load(
        open(config.dev_ref_text_path, "r", encoding="utf-8")
    ))
    final_score = rescore(best_weight, test_am, test_lm)
    predict_hyps = get_highest_score_hyp(final_score, test_hyps)
    cer = cer(test_ref, predict_hyps)
    print("test cer: ", cer)