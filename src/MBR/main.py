import sys
sys.path.append("../../src")
from typing import Callable, List

import torch
import numpy as np
import bert_score
from jiwer import cer

from util.arg_parser import ArgParser
from util.parse_json import parse_json


def find_best_length(n_best: int, dev_refs: List[str], dev_hyps: List[List[str]]):
    cer_record = []
    best_cer = sys.maxsize
    best_length = 2
    for length in range(2, n_best+1):
        cands = []
        refs = []
        for utt_hyps in dev_hyps:
            for hyp_i_pos in range(length):
                hyp_i = [utt_hyps[hyp_i_pos]] * (length - 1)
                other_hyps = utt_hyps[:hyp_i_pos] + utt_hyps[hyp_i_pos+1:length]
                cands += hyp_i
                refs += other_hyps

        # recall size: utt_num * length * (length - 1)
        _, recall, __ = bert_score.score(
            cands,
            refs,
            lang="zh",
            verbose=True,
            device="gpu",
            batch_size=64
        )
        recall = torch.reshape(recall, (len(dev_hyps), length, length-1))
        recall = recall.sum(dim=-1)
        max_value_index = recall.argmax(dim=-1).cpu()

        prediction = []
        for utt_id, v_index in enumerate(max_value_index):
            prediction.append(dev_hyps[utt_id][v_index])
        current_cer = cer(dev_refs, prediction)
        if current_cer < best_cer:
            best_cer = current_cer
            best_length = length
    return cer_record, best_cer, best_length


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    dev_refs, dev_hyps = parse_json(
        file_path=config.dev_file,
        requirements=["ref_text", "hyp_text"], 
        max_utts=config.max_utts,
        n_best=config.n_best
    )

    test_hyps = parse_json(
        file_path=config.test_file,
        requirements=["hyp_text"], 
        max_utts=config.max_utts,
        n_best=config.n_best
    )

    cer_record, best_cer, best_length = find_best_length(config.n_best, dev_refs, dev_hyps)
    print("cer_record:", cer_record)
    print("best_cer:", best_cer)
    print("best_length:", best_length)