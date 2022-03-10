import json
from typing import List

def flatten_2dlist(input_list):
    output_list = []
    for l in input_list:
        output_list += l
    return output_list


def parse_json(
    file_path: str,
    requirements: List[str],
    max_utts: int = -1, 
    n_best: int = -1,
    flatten: bool = False):
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
        n_best: How many hypothesis in an utterance you want to parse.
            -1 means parse all hypothesis.
        flatten: Whether to flatten each the return result.

    Returns:

    """

    output = {}

    json_data = json.load(open(file_path, "r", encoding="utf-8"))

    if "all" in requirements:
        output["all"] = {}
        for utt_count, (utt_id, utt_content) in enumerate(json_data.items(), 1):
            if utt_count > max_utts and max_utts != -1:
                break
            output["all"][utt_id] = {}
            output["all"][utt_id]["ref"] = utt_content["ref"]
            output["all"][utt_id]["hyp"] = {
                hyp_id: hyp_content
                for index, (hyp_id, hyp_content) in enumerate(utt_content["hyp"].items())
                if  index < n_best or n_best == -1
            }

    json_data = json_data.values()

    if "ref_text" in requirements:
        ref_text = [utt["ref"] 
                    for utt_count, utt in enumerate(json_data, 1)
                    if utt_count <= max_utts or max_utts == -1]
        output["ref_text"] = ref_text

    all_hyps = [utt["hyp"]
                for utt_count, utt in enumerate(json_data, 1)
                if utt_count <= max_utts or max_utts == -1]
    
    if "hyp_text" in requirements:
        hyp_text = [
            [hyp["text"] for index, hyp in enumerate(utt_hyps.values()) if index < n_best or n_best == -1]
            for utt_hyps in all_hyps
        ]
        output["hyp_text"] = flatten_2dlist(hyp_text) if flatten else hyp_text

    if "hyp_score" in requirements:
        hyp_score = [
            [hyp["score"] for index, hyp in enumerate(utt_hyps.values()) if index < n_best or n_best == -1]
            for utt_hyps in all_hyps
        ]
        output["hyp_score"] = flatten_2dlist(hyp_score) if flatten else hyp_score

    if "hyp_cer" in requirements:
        hyp_cer = [
            [hyp["cer"] for index, hyp in enumerate(utt_hyps.values()) if index < n_best or n_best == -1]
            for utt_hyps in all_hyps
        ]
        output["hyp_cer"] = flatten_2dlist(hyp_cer) if flatten else hyp_cer
    
    if "alignment" in requirements:
        alignment = [
            [hyp["alignment"] for index, hyp in enumerate(utt_hyps.values()) if index < n_best or n_best == -1]
            for utt_hyps in all_hyps
        ]
        output["alignment"] = alignment

    if len(requirements) > 1:
        return [output[requirement] for requirement in requirements]
    else:
        return output[requirements[0]]