import sys
from typing import Dict, List

from sklearn.metrics import max_error
sys.path.append("..")
import json
from tqdm import tqdm
from jiwer import cer
from transformers import BertTokenizer 
from espnet_data.preprocess.align import levenshtein_distance_alignment 

def merge_alignments(alignment_i, alignment_j):
    print("i: ", alignment_i)
    print("j: ", alignment_j)

    i, j, merged_alignment= 0, 0, []

    while i < len(alignment_i) and j < len(alignment_j):
        if alignment_i[i][0] == alignment_j[j][0]:
            align_element = [alignment_i[i][0], alignment_i[i][1], alignment_j[j][1]]
            i += 1
            j += 1
        else:
            if alignment_i[i][0] == "-":
                align_element = ["-", alignment_i[i][1], "-"]
                i += 1
            else:
                align_element = ["-", "-", alignment_j[j][1]]
                j += 1
        merged_alignment.append(align_element)
    while i < len(alignment_i):
        merged_alignment.append(["-", alignment_i[i][1], "-"])
        i += 1
    while j < len(alignment_j):
        merged_alignment.append(["-", "-", alignment_j[j][1]])
        j += 1

    return merged_alignment


def new_merge_alignments(alignment_i, alignment_j):
    i, j, merged_alignment= 0, 0, []

    while i < len(alignment_i) and j < len(alignment_j):
        if alignment_i[i][0] == alignment_j[j][0]:
            align_element = alignment_i[i] + alignment_j[j][1:]
            i += 1
            j += 1
        else:
            if alignment_i[i][0] == "*":
                align_element = alignment_i[i] + alignment_j[j][1:]
                i += 1
            else:
                align_element = ["*"]*len(alignment_i[i]) + alignment_j[j][1:]
                j += 1
        merged_alignment.append(align_element)
    while i < len(alignment_i):
        merged_alignment.append(alignment_i[i] + ["*"]*len(alignment_j[0][1:]))
        i += 1
    while j < len(alignment_j):
        merged_alignment.append(["*"]*len(alignment_i[0]) + alignment_j[j][1:])
        j += 1

    return merged_alignment


def get_feature(config, data_paths, require_features):

    bert_tokenizer = BertTokenizer.from_pretrained(config.model.bert)
    feature_jsons = {}
    for path, feature in zip(data_paths, require_features):
        feature_jsons[feature] = json.load(open(path, "r", encoding="utf-8"))

    # initialize the output data format
    output = []        
    for utt_num, (utt_id, _) in enumerate(feature_jsons["ref_text"].items()):
        if utt_num == config.max_utt: break
        output.append({"utt_id": utt_id})

    for row in tqdm(output):
        utt_id = row["utt_id"]

        other_token_seqs = []
        for hyp_num, hyp_text in enumerate(feature_jsons["hyp_text"][utt_id].values()):
            if hyp_num == config.n_best: break
            hyp_seq = bert_tokenizer.tokenize(hyp_text)
            if hyp_num == 0:
                top_one_token_seq = hyp_seq
            else:
                other_token_seqs.append(hyp_seq)

        alignments = []
        for other_token_seq in other_token_seqs: 
            alignment = levenshtein_distance_alignment(
                top_one_token_seq,
                other_token_seq
            )
            alignment = [[ref_token, hyp_token] 
                for ref_token, hyp_token in zip(alignment[0], alignment[1])
            ]
            alignments.append(alignment)

        merged_alignment = alignments.pop(0)
        for alignment in alignments:
            merged_alignment = new_merge_alignments(
                merged_alignment,
                alignment
            )

#################################################################

        possible_token_sets = [list(set(token_pair)) for token_pair in merged_alignment]
        combinations = [[token] for token in possible_token_sets[0]]
        paths = [[num] for num in range(len(possible_token_sets[0]))]
        for token_set in possible_token_sets[1:]:
            new_combinations = []
            new_paths = []
            for choice_id, token in enumerate(token_set):
                for seq, path in zip(combinations, paths):
                    new_combinations.append(seq + [token])
                    new_paths.append(path + [choice_id])
            combinations = new_combinations
            paths = new_paths


        ref = feature_jsons["ref_text"][utt_id]
        min_cer = sys.maxsize
        for choice_seq in combinations:
            blank_removed_seq = \
                "".join([token for token in choice_seq if token != "*"])

            error_rate = cer(ref, blank_removed_seq)
            if error_rate < min_cer:
                min_cer = error_rate
                target_seq = choice_seq

        label = []
        for token, tokens in zip(target_seq, merged_alignment):
            label.append(tokens.index(token))

#################################################################
        input_ids = []
        token_type_ids  = []
        prediction_pos = []
        for idx, token_pair in enumerate(merged_alignment):
            prediction_pos += [len(input_ids)]
            input_ids += (["[CLS]"] + token_pair) if idx == 0 else (["[SEP]"] + token_pair)
            token_type = [0] if idx % 2 == 0 else [1]
            token_type_ids += token_type * (len(token_pair) + 1)
        input_ids = bert_tokenizer.convert_tokens_to_ids(input_ids)
        row.update({
            "input_ids": input_ids,
            "attention_masks": [1] * (len(input_ids)),
            "token_type_ids": token_type_ids,
            "prediction_pos": prediction_pos,
            "labels": label
        })

    return output


if __name__ == "__main__":
    a_i = levenshtein_distance_alignment(["1", "2", "3", "4"], ["1", "2", "3"])
    a_i = [ [ref_token, hyp_token] for ref_token, hyp_token in zip(a_i[0], a_i[1]) ]

    a_j = levenshtein_distance_alignment(["1", "2", "3", "4"], ["1", "2", "3", "4", "5"])
    a_j = [ [ref_token, hyp_token] for ref_token, hyp_token in zip(a_j[0], a_j[1]) ]

    a_k = levenshtein_distance_alignment(["1", "2", "3", "4"], ["1", "3"])
    a_k = [ [ref_token, hyp_token] for ref_token, hyp_token in zip(a_k[0], a_k[1]) ]


    new_alignment = new_merge_alignments(a_i, a_j)
    new_alignment = new_merge_alignments(new_alignment, a_k)