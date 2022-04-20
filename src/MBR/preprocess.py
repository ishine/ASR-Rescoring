import csv
import json
from torch import nn
from tqdm import tqdm

# espnet asr recognition result path
input_root = "/home/chkuo/chkuo/experiment/bertscore_MBR/train_sp_pytorch_train"
train_path = input_root + "/decode_train_sp_decode_lm_4/data.json"
dev_path = input_root + "/decode_dev_decode_lm_4/data.json"
test_path = input_root + "/decode_test_decode_lm_4/data.json"
inputs = [train_path, dev_path, test_path]


# csv output files path 
output_root = "/home/chkuo/chkuo/experiment/bertscore_MBR/data"
train_dataset_path = output_root + "/AISHELL1_train.csv"
dev_dataset_path = output_root + "/AISHELL1_dev.csv"
test_dataset_path = output_root + "/AISHELL1_test.csv"
outputs = [train_dataset_path, dev_dataset_path, test_dataset_path]

for input_index, input in enumerate(inputs):
    with open(input, "r") as in_file:
        with open(outputs[input_index], "w", newline = "") as out_file:
            writer = csv.writer(out_file)
            header = \
            ["reference"] + \
            ["candidate_{}".format(index) for index in range(10)] + \
            ["score_{}".format(index) for index in range(10)]
            writer.writerow(header)
            
            data = json.load(in_file)
            utt_list = [utt for utt in data["utts"]]

            for utt in tqdm(utt_list):
                candidate_list = data["utts"][utt]["output"]
                
                # 取得 reference sentence
                reference_sen = candidate_list[0]["text"]
                
                # 取得 candidate sentences 與 scores
                candidate_sens = [None] * 10
                candidate_scores = [None] * 10
                for candidate_index, candidate in enumerate(candidate_list):
                    candidate_sens[candidate_index] = candidate["rec_text"].strip("<eos>")
                    candidate_scores[candidate_index] = candidate["score"]
                    
                writer.writerow([reference_sen] + candidate_sens + candidate_scores)