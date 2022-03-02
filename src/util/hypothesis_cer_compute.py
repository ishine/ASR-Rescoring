import csv
import json
from torch import nn
from tqdm import tqdm
from jiwer import cer

# json files paths 
root = "D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\data\\"
train_dataset_path = root + "train.am.json"
dev_dataset_path = root + "dev.am.json"
test_dataset_path = root + "test.am.json"
#data_paths = [train_dataset_path, dev_dataset_path, test_dataset_path]
data_paths = [test_dataset_path]

for data_path in data_paths:
    with open(data_path, "r", encoding="utf-8") as in_file:
        json_data = json.load(in_file)

    for utt_id, recog_content in tqdm(json_data.items()):
        for sentence_id, sentence_content in recog_content.items():
            if sentence_id == "ref":
                reference = sentence_content
            else:
                hypothesis = sentence_content["text"]
                character_error_rate = cer(reference, hypothesis)
                sentence_content["cer"] = character_error_rate

    with open(data_path, "w", encoding="utf8") as out_file:
        json.dump(json_data, out_file, ensure_ascii=False, indent=4)
