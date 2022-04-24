import sys
from typing import Dict, List
sys.path.append("..")
import json
from tqdm import tqdm
from transformers import BertTokenizer 

def get_feature(config, data_paths, require_features):

    bert_tokenizer = BertTokenizer.from_pretrained(config.model.bert)

    feature_set = {}
    for path, feature in zip(data_paths, require_features):
        feature_set[feature] = json.load(open(path, "r", encoding="utf-8"))

    # initialize the output data format
    output = []
    for id, feature in enumerate(feature_set.values()):
        if id == 1: break
        for utt_count, (utt_id, hyps) in enumerate(feature.items()):
            if utt_count == config.max_utt: break
            for hyp_count, (hyp_id, _) in enumerate(hyps.items()):
                if hyp_count == config.n_best: break
                output.append({"utt_id": utt_id, "hyp_id": hyp_id})

    for row in tqdm(output):
        for feature, feature_json in feature_set.items():
            utt_id = row["utt_id"]
            hyp_id = row["hyp_id"]

            if feature == "hyps_token_ids":
                token_seq = bert_tokenizer.tokenize(feature_json[utt_id][hyp_id])
                row.update({
                    "hyps_token_ids": 
                        bert_tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        ),
                    "attention_masks": [1] * (len(token_seq) + 2)
                })

            elif feature == "hyps_am_score":
                row.update({
                        "hyps_am_score": feature_json[utt_id][hyp_id]
                })

            elif feature == "hyps_cer":
                row.update({
                        "hyps_cer": feature_json[utt_id][hyp_id]
                })

            elif feature == "mlm_pll_score":
                row.update({
                        "mlm_pll_score": feature_json[utt_id][hyp_id]
                })

    return output