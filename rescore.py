import sys
import json
import logging
import inspect
from typing import Dict

import numpy as np
from jiwer import cer 
from tqdm import tqdm

from util.arg_parser import ArgParser

def dict_to_list(dict):
    all_scores = []
    for utt_id, hyps in dict.items():
        if isinstance(hyps, Dict):
            row_scores = []
            for hyp_id, hyp in hyps.items():
                row_scores.append(hyp)
            all_scores.append(row_scores)
        else:
            all_scores.append(hyps)
    return all_scores

def find_best_weight(am, lm, hyps, ref):
    best_cer = sys.float_info.max

    hyps_len = []
    for utt_hyps in hyps:
        utt_hyps_len = []
        for hyp in utt_hyps:
            utt_hyps_len.append(len(hyp))
        hyps_len.append(utt_hyps_len)

    for weight in tqdm(np.arange(0.00, 1.0, 0.01)):
        final_score = rescore(weight, hyps_len, am, lm)
        predict_hyps = get_highest_score_hyp(final_score, hyps)
        error = cer(ref, predict_hyps)
        if error < best_cer:
            best_cer = error
            best_weight = weight

    return best_weight, best_cer

def rescore(weight, hyps_len, am, lm):
    am = np.array(am)
    lm = np.array(lm)
    hyps_len = np.array(hyps_len)
    final_score = (1-weight)*(am)/hyps_len + weight*(lm)/hyps_len
    return final_score

def get_highest_score_hyp(final_score, hyps):
    max_score_hyp_index = np.argmax(final_score, axis=-1)
    best_hyp = [ht[index] for ht, index in zip(hyps, max_score_hyp_index)]
    return best_hyp


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    logging.basicConfig(
        filename=config.output_path + "/rescore.log",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logging.info(config)
    logging.info("\n" + inspect.getsource(find_best_weight))
    logging.info("\n" + inspect.getsource(rescore))

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
    logging.info("best_weight: " + str(best_weight))
    logging.info("dev cer: " + str(best_cer))
    print("best_weight: ", best_weight)
    print("dev cer: ", best_cer)

    test_am = dict_to_list(json.load(
        open(config.test_am_path, "r", encoding="utf-8")
    ))
    test_lm = dict_to_list(json.load(
        open(config.test_lm_path, "r", encoding="utf-8")
    ))
    test_hyps = dict_to_list(json.load(
        open(config.test_hyps_text_path, "r", encoding="utf-8")
    ))
    test_ref = dict_to_list(json.load(
        open(config.test_ref_text_path, "r", encoding="utf-8")
    ))

    hyps_len = []
    for utt_hyps in test_hyps:
        utt_hyps_len = []
        for hyp in utt_hyps:
            utt_hyps_len.append(len(hyp))
        hyps_len.append(utt_hyps_len)

    final_score = rescore(best_weight, hyps_len, test_am, test_lm)
    predict_hyps = get_highest_score_hyp(final_score, test_hyps)
    cer = cer(test_ref, predict_hyps)
    logging.info("test cer: " + str(cer))
    print("test cer: ", cer)

