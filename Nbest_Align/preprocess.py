import sys
from typing import Dict, List
sys.path.append("..")
import json
from tqdm import tqdm
from transformers import BertTokenizer 
from espnet_data.preprocess.align import levenshtein_distance_alignment 

def merge_alignments(alignment_i, alignment_j):
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


def get_feature(config, data_paths, require_features):

    bert_tokenizer = BertTokenizer.from_pretrained(config.model.bert)
    feature_set = {}
    for path, feature in zip(data_paths, require_features):
        feature_set[feature] = json.load(open(path, "r", encoding="utf-8"))


    # initialize the output data format
    output = []
    for id, feature in enumerate(feature_set.values()):
        if id == 1: break
        
        for utt_num, (utt_id, _) in enumerate(feature.items()):
            if utt_num == config.max_utt: break
            output.append({"utt_id": utt_id})


    for row in tqdm(output):
        for feature, feature_json in feature_set.items():
            utt_id = row["utt_id"]

            if feature == "ref_token_ids":
                ref_seq = bert_tokenizer.tokenize(feature_json[utt_id])
                ref_token_ids = bert_tokenizer.convert_tokens_to_ids(ref_seq)
                row.update({
                    "ref_token_ids": ref_token_ids
                })

            if feature == "hyps_token_ids":
                other_token_seqs = []
                for hyp_num, hyp_text in enumerate(feature_json[utt_id].values()):
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
                    merged_alignment = merge_alignments(
                        merged_alignment,
                        alignment
                    )

                hyps_token_ids = []
                token_type_ids  = []
                cls_pos = []
                for idx, token_pair in enumerate(merged_alignment):
                    cls_pos += [len(hyps_token_ids)]
                    hyps_token_ids += ["[CLS]"] + token_pair
                    token_type = [0] if idx % 2 == 0 else [1]
                    token_type_ids += token_type * len(["[CLS]"] + token_pair)
                hyps_token_ids = bert_tokenizer.convert_tokens_to_ids(hyps_token_ids)
                
                row.update({
                    "hyps_token_ids": hyps_token_ids,
                    "attention_masks": [1] * (len(hyps_token_ids)),
                    "token_type_ids": token_type_ids,
                    "cls_pos": cls_pos
                })

            '''
            elif feature == "hyps_am_scores":
                hyps_am_scores = []
                for hyp_num, hyp_am_score in enumerate(feature_json[utt_id].values()):
                    if hyp_num == config.n_best: break
                    hyps_am_scores.append(hyp_am_score)
                row.update({
                        "hyps_am_scores": hyps_am_scores
                })
            '''
    return output

if __name__ == "__main__":
    a_i = levenshtein_distance_alignment(["how", "are", "you", "do"], ["how", "you", "doing"])
    a_i = [ [ref_token, hyp_token] for ref_token, hyp_token in zip(a_i[0], a_i[1]) ]

    a_j = levenshtein_distance_alignment(["how", "are", "you", "do"], ["who", "are", "you", "doing"])
    a_j = [ [ref_token, hyp_token] for ref_token, hyp_token in zip(a_j[0], a_j[1]) ]

    print(merge_alignments(a_i, a_j))
