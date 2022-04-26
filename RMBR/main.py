import sys
sys.path.append("..")
from typing import List
import logging

from jiwer import cer

from mbr import mbr_decode
from utility_functions import BaseFunction, BertScoreFunction, CerScoreFunction
from preprocess import get_feature
from util.arg_parser import ArgParser
from util.saving import json_saving
from util.get_output_format import get_output_format

def find_best_length(
        n_best: int,
        dev_refs: List[str],
        dev_hyps: List[List[str]],
        utility_function: BaseFunction
    ):

    best_cer = sys.maxsize
    best_length = 2
    best_mbr_score = []
    for length in range(2, n_best+1):
        prediction, score = mbr_decode(length, dev_hyps, utility_function)
        current_cer = cer(dev_refs, prediction)
        logging.info(f"Use top-{length} candidate list, cer: {current_cer}")
        print(f"Use top-{length} candidate list, cer: {current_cer}")
        if current_cer < best_cer:
            best_cer = current_cer
            best_length = length
            best_mbr_score = score

    return best_cer, best_length, best_mbr_score


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()
    
    logging.basicConfig(
        filename=config.output_path + "/mbr.log",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    dev_refs, dev_hyps = get_feature(
        config,
        data_paths=config.dev_feature_path,
        require_features=config.dev_feature
    )

    test_refs, test_hyps = get_feature(
        config,
        data_paths=config.test_feature_path,
        require_features=config.test_feature
    )

    if config.utility_function == "bertscore":
        utility_function = BertScoreFunction(config)
    elif config.utility_function == "cer":
        utility_function = CerScoreFunction(config)

    logging.info(f"Running MBR on dev set to find best length ...")
    print("Running MBR on dev set to find best length ...")
    best_cer, best_length, best_mbr_score = find_best_length(
        config.n_best,
        dev_refs,
        dev_hyps,
        utility_function
    )
    logging.info(f"best_cer: {best_cer}")
    logging.info(f"best_length: {best_length}")
    print("best_cer: ", best_cer)
    print("best_length: ", best_length)

    output_json = get_output_format(
        config.dev_output_format,
        config.max_utt,
        config.n_best
    )
    best_mbr_score = best_mbr_score.tolist()
    for (utt_id, utt_content), utt_scores in zip(output_json.items(), best_mbr_score):
        for (hyp_id, hyp_content), hyp_score in zip(utt_content.items(), utt_scores):
            output_json[utt_id][hyp_id] = hyp_score
    json_saving(config.output_path + "/dev_MBR.json", output_json)
    

    logging.info(f"Running MBR on test set ...")
    print("Running MBR on test set ...")
    prediction, best_mbr_score = mbr_decode(best_length, test_hyps, utility_function)
    test_cer = cer(test_refs, prediction)
    logging.info(f"test cer: {test_cer}")
    print("test cer: ", cer(test_refs, prediction))

    output_json = get_output_format(
        config.test_output_format,
        config.max_utt,
        config.n_best
    )
    best_mbr_score = best_mbr_score.tolist()
    for (utt_id, utt_content), utt_scores in zip(output_json.items(), best_mbr_score):
        for (hyp_id, hyp_content), hyp_score in zip(utt_content.items(), utt_scores):
            output_json[utt_id][hyp_id] = hyp_score
    json_saving(config.output_path + "/test_MBR.json", output_json)