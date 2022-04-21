import json
from tqdm import tqdm
from transformers import BertTokenizer 

from util.saving import json_saving

if __name__ == "__main__":
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    input_paths = {
        "train": "../espnet_data/train/ref_text.json",
        "dev": "../espnet_data/dev/ref_text.json",
        "test": "../espnet_data/test/ref_text.json"
    }

    output_paths = {
        "train": "preprocessed_data/train.json",
        "dev": "preprocessed_data/dev.json",
        "test": "preprocessed_data/test.json"
    }

    output_jsons = {
        "train": {},
        "dev": {},
        "test": {}
    }
    
    for data_type, file_path in input_paths.items():
        json_data = json.load(open(file_path, "r", encoding="utf-8"))
        for utt_id, ref_text in tqdm(json_data.items()):

            output_jsons[data_type][utt_id] = {}
            output_paths[data_type][utt_id]["input_ids"] = []
            if data_type != "test":
                output_paths[data_type][utt_id]["labels"] = []
            output_paths[data_type][utt_id]["attention_masks"] = []

            token_seq = bert_tokenizer.tokenize(ref_text)
            for mask_pos in range(len(token_seq)):
                output_jsons[data_type]["input_ids"].append(
                    bert_tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + token_seq[:mask_pos]
                        + ["[MASK]"] + token_seq[mask_pos+1:]
                        + ["[SEP]"]
                    )
                )

                if data_type != "test":
                    output_jsons[data_type]["labels"].append(
                        bert_tokenizer.convert_token_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )

                output_jsons[data_type]["attention_masks"] = [1] * (len(token_seq) + 2)

    # save preprocessed data
    for data_type, file_path in output_paths.items():
        json_saving(
            file_path,
            output_jsons[data_type]
        )