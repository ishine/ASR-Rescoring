import json
from tqdm import tqdm
from transformers import BertTokenizer


def get_feature(config, data_paths, require_features):

    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    feature_set = {}
    for path, feature in zip(data_paths, require_features):
        feature_set[feature] = json.load(open(path, "r", encoding="utf-8"))


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

    return output