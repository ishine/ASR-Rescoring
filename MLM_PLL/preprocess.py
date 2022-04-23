import sys
sys.path.append("..")
import json
from tqdm import tqdm
from transformers import BertTokenizer 

from util.saving import json_saving

def do_job(sentence, utt_id, hyp_id, task_type, output_json):
    token_seq = bert_tokenizer.tokenize(sentence)
    for mask_pos in range(len(token_seq)):
        one_data = {}
        one_data["utt_id"] = utt_id
        one_data["hyp_id"] = hyp_id
        one_data["input_ids"] = (
            bert_tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + token_seq[:mask_pos]
                + ["[MASK]"] + token_seq[mask_pos+1:]
                + ["[SEP]"]
            )
        )
        one_data["attention_masks"] = [1] * (len(token_seq) + 2)    #  "+2" for [CLS] and [SEP]
        one_data["mask_pos"] = mask_pos + 1     #  "+1" for [CLS]
        one_data["labels"] = (
            bert_tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + token_seq + ["[SEP]"]
            )
        )
        output_json.append(one_data)
    return output_json


if __name__ == "__main__":
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    jobs = [{
            "task": "for_training", 
            "in": "../espnet_data/alfred/train/ref_text.json",
            "out": "preprocessed_data/for_training/train.json"
        }, {
            "task": "for_training",
            "in": "../espnet_data/alfred/dev/ref_text.json",
            "out": "preprocessed_data/for_training/dev.json"
        }, {
            "task": "for_scoring",
            "in": "../espnet_data/alfred/train/hyps_text.json",
            "out": "preprocessed_data/for_scoring/train.json"
        }, {
            "task": "for_scoring",
            "in": "../espnet_data/alfred/dev/hyps_text.json",
            "out": "preprocessed_data/for_scoring/dev.json"
        }, {
            "task": "for_scoring",
            "in": "../espnet_data/alfred/test/hyps_text.json",
            "out": "preprocessed_data/for_scoring/test.json"
        }
    ]
    
    for job in jobs:
        json_data = json.load(open(job["in"], "r", encoding="utf-8"))

        output_json = []
        if job["task"] == "for_training":
            for utt_id, sentence in tqdm(json_data.items()):
                output_json = do_job(sentence, utt_id, None, job["task"], output_json)

        elif job["task"] == "for_scoring":
            for utt_id, hyps in tqdm(json_data.items()):
                for hyp_id, sentence in hyps.items():
                    output_json = do_job(sentence, utt_id, hyp_id, job["task"], output_json)

        json_saving(job["out"], output_json)