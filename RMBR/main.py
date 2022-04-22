import sys
sys.path.append("..")
from typing import List
import logging

from jiwer import cer

from mbr import mbr_decode
from utility_functions import BaseFunction, BertScoreFunction
from util.arg_parser import ArgParser
from util.parse_json import parse_json
from util.saving import json_saving


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

    dev_refs, dev_hyps = parse_json(
        file_path=config.dev_file,
        requirements=["ref_text", "hyp_text"], 
        max_utts=config.max_utts,
        n_best=config.n_best
    )

    test_refs, test_hyps = parse_json(
        file_path=config.test_file,
        requirements=["ref_text", "hyp_text"], 
        max_utts=config.max_utts,
        n_best=config.n_best
    )

    if config.utility_function == "bertscore":
        utility_function = BertScoreFunction(config)
    
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

    output_json = parse_json(
        file_path=config.dev_file,
        requirements=["all"],
        max_utts=config.max_utts,
        n_best=config.n_best,
        flatten=False
    )
    best_mbr_score = best_mbr_score.tolist()
    for (utt_id, utt_content), scores in zip(output_json.items(), best_mbr_score):
        for (hyp_id, hyp_content), score in zip(utt_content["hyp"].items(), scores):
            hyp_content["score"] = score
    json_saving(config.output_path + "/dev.MBR.json", output_json)
    

    logging.info(f"Running MBR on test set ...")
    print("Running MBR on test set ...")
    prediction, best_mbr_score = mbr_decode(best_length, test_hyps, utility_function)
    best_mbr_score = best_mbr_score.tolist()
    test_cer = cer(test_refs, prediction)
    logging.info(f"test cer: {test_cer}")
    print("test cer: ", cer(test_refs, prediction))

    output_json = parse_json(
        file_path=config.test_file,
        requirements=["all"],
        max_utts=config.max_utts,
        n_best=config.n_best,
        flatten=False
    )
    for (utt_id, utt_content), scores in zip(output_json.items(), best_mbr_score):
        for (hyp_id, hyp_content), score in zip(utt_content["hyp"].items(), scores):
            hyp_content["score"] = score
    json_saving(config.output_path + "/test.MBR.json", output_json)