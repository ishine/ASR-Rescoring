import csv
import json
from torch import nn
from tqdm import tqdm

# espnet asr recognition result path
input_root = "/home/chkuo/chkuo/experiment/bert_semantic_rescoring/train_sp_pytorch_train"
train_path = input_root + "/decode_train_sp_decode_lm_4/data.json"
dev_path = input_root + "/decode_dev_decode_lm_4/data.json"
test_path = input_root + "/decode_test_decode_lm_4/data.json"
inputs = [train_path, dev_path, test_path]


# json output files path 
output_root = "/home/chkuo/chkuo/experiment/mlm-scoring/examples/asr-aishell1-espnet/data"
train_dataset_path = output_root + "/train.am.json"
dev_dataset_path = output_root + "/dev.am.json"
test_dataset_path = output_root + "/test.am.json"
outputs = [train_dataset_path, dev_dataset_path, test_dataset_path]

n_best = 10

for input_index, input in enumerate(inputs):
    with open(input, "r") as in_file:
        with open(outputs[input_index], "w", newline = "") as out_file:
            
            formatted_data = {}

            data = json.load(in_file)
            utt_list = [utt for utt in data["utts"]]

            for utt in tqdm(utt_list):

                candidate_list = data["utts"][utt]["output"]

                formatted_data[utt] = {}
               
                # 取得 reference sentence
                formatted_data[utt]["ref"] = candidate_list[0]["text"]
                
                # 取得 hypothesis sentences 的 scores 與 text
                for candidate_index in range(n_best):
                    formatted_data[utt]["hyp_{}".format(candidate_index+1)] = {}

                    formatted_data[utt]["hyp_{}".format(candidate_index+1)]["score"] = \
                        candidate_list[candidate_index]["score"]

                    formatted_data[utt]["hyp_{}".format(candidate_index+1)]["text"] = \
                        candidate_list[candidate_index]["rec_text"].strip("<eos>")

            #json_string = json.dumps(formatted_data, ensure_ascii=False)
            json.dump(formatted_data, out_file, ensure_ascii=False, indent=4)
            #out_file.write(json_string)