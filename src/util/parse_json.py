import json
from typing import List
import unittest


def flatten_2dlist(input_list):
    output_list = []
    for l in input_list:
        output_list += l
    return output_list


def parse_json(file_path: str, requirements: List[str], max_utts: int = -1, flatten: bool = False):
    """
    Parse a specify number of utterance in the json file 
    and return the required components in each utterance.
    
    Args:
        file_path: The path of the json file you want to parse.
        requirements: The item you want to get from the json file.
            requirements can be "all", "ref_text", "hyp_text", "hyp_score"
            , "hyp_cer" or "alignment" 
        max_utts: How many utterances you want to parse.
            -1 means parse all utterances.
        flatten: Whether to flatten each the return result.

    Returns:

    """

    output = {}

    json_data = json.load(open(file_path, "r", encoding="utf-8"))

    if "all" in requirements:
        output["all"] = {utt_id: utt_content 
                         for utt_count, (utt_id, utt_content) in enumerate(json_data.items(), 1)
                         if utt_count <= max_utts}

    json_data = json_data.values()

    if "ref_text" in requirements:
        ref_text = [utt["ref"] 
                    for utt_count, utt in enumerate(json_data, 1)
                    if utt_count <= max_utts]
        output["ref_text"] = ref_text

    all_hyps = [utt["hyp"]
                for utt_count, utt in enumerate(json_data, 1)
                if utt_count <= max_utts]
    
    if "hyp_text" in requirements:
        hyp_text = [
            [hyp["text"] for hyp in utt_hyps.values()]
            for utt_hyps in all_hyps
        ]
        output["hyp_text"] = flatten_2dlist(hyp_text) if flatten else hyp_text

    if "hyp_score" in requirements:
        hyp_score = [
            [hyp["score"] for hyp in utt_hyps.values()]
            for utt_hyps in all_hyps
        ]
        output["hyp_score"] = flatten_2dlist(hyp_score) if flatten else hyp_score

    if "hyp_cer" in requirements:
        hyp_cer = [
            [hyp["cer"] for hyp in utt_hyps.values()]
            for utt_hyps in all_hyps
        ]
        output["hyp_cer"] = flatten_2dlist(hyp_cer) if flatten else hyp_cer
    
    if "alignment" in requirements:
        alignment = [
            [hyp["alignment"] for hyp in utt_hyps.values()]
            for utt_hyps in all_hyps
        ]
        output["alignment"] = alignment

    return [output[requirement] for requirement in requirements]