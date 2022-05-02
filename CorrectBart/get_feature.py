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

    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    feature_set = {}
    for path, feature in zip(data_paths, require_features):
        feature_set[feature] = json.load(open(path, "r", encoding="utf-8"))

    if config.method == "one_hyp":
        # initialize the output data format
        output = []
        feature = feature_set["hyps_token_ids"]
        for utt_num, (utt_id, utt_content) in enumerate(feature.items()):
            if utt_num == config.max_utt: break
            
            for hyp_num, (hyp_id, _) in enumerate(utt_content.items()):
                if hyp_num == config.n_best: break
                output.append({"utt_id": utt_id, "hyp_id": hyp_id})


        for row in tqdm(output):
            for feature, feature_json in feature_set.items():
                utt_id = row["utt_id"]
                hyp_id = row["hyp_id"]

                if feature == "ref_token_ids":
                    ref_seq = tokenizer.tokenize(feature_json[utt_id])
                    ref_token_ids = tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + ref_seq + ["[SEP]"]
                    )
                    row.update({
                        "ref_token_ids": ref_token_ids
                    })

                if feature == "hyps_token_ids":
                    hyp_seq = tokenizer.tokenize(feature_json[utt_id][hyp_id])
                    hyp_token_ids = tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + hyp_seq + ["[SEP]"]
                    )
                    
                    row.update({
                        "hyps_token_ids": hyp_token_ids,
                        "attention_masks": [1] * (len(hyp_token_ids))
                    })

    elif config.method == "n_best_align":
        # initialize the output data format
        output = []
        feature = feature_set["hyps_token_ids"]
        for utt_num, utt_id in enumerate(feature.keys()):
            if utt_num == config.max_utt: break
            output.append({"utt_id": utt_id})


        for row in tqdm(output):
            for feature, feature_json in feature_set.items():
                utt_id = row["utt_id"]

                if feature == "ref_token_ids":
                    ref_seq = tokenizer.tokenize(feature_json[utt_id])
                    ref_token_ids = tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + ref_seq + ["[SEP]"]
                    )
                    row.update({
                        "ref_token_ids": ref_token_ids
                    })

                if feature == "hyps_token_ids":
                    other_token_seqs = []
                    for hyp_num, hyp in enumerate(feature_json[utt_id].values()):
                        if hyp_num == config.n_best: break

                        hyp_seq = ["[CLS]"] + tokenizer.tokenize(hyp) + ["[SEP]"]
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

                    input_token_ids = []
                    for alignment in merged_alignment:
                        input_token_ids.append(tokenizer.convert_tokens_to_ids(alignment))

                    row.update({
                        "hyps_token_ids": input_token_ids,
                        "attention_masks": [1] * (len(input_token_ids)),
                    })

    return output