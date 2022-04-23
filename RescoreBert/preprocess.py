import sys
sys.path.append("..")
import json
from tqdm import tqdm
from transformers import BertTokenizer 

from util.saving import json_saving

def get_feature(raw_data, require_feature):
    output = []
    
    for utt_id, hyps in tqdm(raw_data.items()):
        for hyp_id, hyp_content in hyps.items():

            single_data = {"utt_id": utt_id, "hyp_id": hyp_id}

            if require_feature == "hyps_token_ids":
                token_seq = bert_tokenizer.tokenize(hyp_content)
                single_data["hyps_token_ids"] = \
                    bert_tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + token_seq + ["[SEP]"]
                    )

            elif require_feature == "hyps_am_score":
                single_data["hyps_am_score"] = hyp_content

            elif require_feature == "hyps_cer":
                single_data["hyps_cer"] = hyp_content

            elif require_feature == "mlm_pll_score":
                single_data["mlm_pll_score"] = hyp_content

            output.append(single_data)

    return output


if __name__ == "__main__":
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    jobs = [{
            "task": "for_training", 
            "feature_in_path": [
                "../espnet_data/alfred/train/hyps_text.json",
                "../espnet_data/alfred/train/hyps_score.json"
            ],
            "require_features": [
                "hyps_token_ids",
                "hyps_am_score"
            ],
            "label_in_path": [
                "../MLM_PLL/result",
                "../espnet_data/alfred/train/hyps_cer.json"
            ],
            "label": [
                "mlm_pll_score",
                "hyps_cer"
            ],
            "out": "preprocessed_data/for_training/train"
        }, {
            "task": "for_training",
            "feature_in_path": [
                "../espnet_data/alfred/dev/hyps_text.json",
                "../espnet_data/alfred/dev/hyps_score.json"
            ],
            "require_features": [
                "hyps_token_ids",
                "hyps_am_score"
            ],
            "label_in_path": [
                "../MLM_PLL/result",
                "../espnet_data/alfred/dev/hyps_cer.json"
            ],
            "label": [
                "mlm_pll_score",
                "hyps_cer"
            ],
            "out": "preprocessed_data/for_training/dev"
        }, {
            "task": "for_scoring",
            "feature_in_path": [
                "../espnet_data/alfred/dev/hyps_text.json",
                "../espnet_data/alfred/dev/hyps_score.json",
            ],
            "require_features": [
                "hyps_token_ids",
                "hyps_am_score",
            ],
            "out": "preprocessed_data/for_scoring/dev"
        }, {
            "task": "for_scoring",
            "feature_in_path": [
                "../espnet_data/alfred/test/hyps_text.json",
                "../espnet_data/alfred/test/hyps_score.json",
            ],
            "require_features": [
                "hyps_token_ids",
                "hyps_am_score",
            ],
            "out": "preprocessed_data/for_scoring/test"
        }
    ]
    
    for job_id, job in enumerate(jobs):
        print(f"job {job_id}, total: {len(job)}")

        for path, feature in zip(job["feature_in_path"], job["require_features"]):
            raw_data = json.load(open(path, "r", encoding="utf-8"))
            output = get_feature(raw_data, feature)
            json_saving(job["out"] + "/" + feature + ".json", output)

        if job["task"] == "for_training":
            for path, label in zip(job["label_in_path"], job["label"]):
                raw_data = json.load(open(path, "r", encoding="utf-8"))
                output = get_feature(raw_data, label)
                json_saving(job["out"] + "/" + label + ".json", output)