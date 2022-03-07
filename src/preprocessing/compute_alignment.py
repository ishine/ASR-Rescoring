import json
from tqdm import tqdm
from levenshtein import levenshtein_distance_alignment

# espnet asr recognition result path
input_root = "/home/chkuo/chkuo/experiment/bert_semantic_rescoring/train_sp_pytorch_train"
train_path = input_root + "/decode_train_sp_decode_lm_4/data.json"
dev_path = input_root + "/decode_dev_decode_lm_4/data.json"
test_path = input_root + "/decode_test_decode_lm_4/data.json"
inputs = [train_path, dev_path, test_path]

# json output files path 
output_root = "/home/chkuo/chkuo/experiment/ASR-Rescoring/data"
train_dataset_path = output_root + "/train.alignment.json"
dev_dataset_path = output_root + "/dev.alignment.json"
test_dataset_path = output_root + "/test.alignment.json"
outputs = [train_dataset_path, dev_dataset_path, test_dataset_path]

n_best = 10

for input_path, output_path in zip(inputs, outputs):

    with open(input_path, "r") as in_file:
        input_json_data = json.load(in_file)

    utts = {utt_id: utt_content for utt_id, utt_content in input_json_data["utts"].items()}

    output_json_data = {}

    for utt_id, utt_content in tqdm(utts.items()):

        candidate_list = utt_content["output"]

        # 取得 reference sentence
        ref = candidate_list[0]["text"]
        
        output_json_data[utt_id] = {}
        output_json_data[utt_id]["ref"] = ref
        output_json_data[utt_id]["hyp"] = {}

        # 取得 hypothesis sentences 計算 ref 和 hyp 的 alignment
        for candidate_index in range(n_best):
            hyp = candidate_list[candidate_index]["rec_text"].strip("<eos>")
            
            alignment = levenshtein_distance_alignment(hyp, ref)
            
            ref_tokens, hyp_tokens, operation_tokens = "", "", ""
            for (ref_token, hyp_token, operation_token) in alignment:
                ref_tokens += ref_token
                hyp_tokens += hyp_token
                operation_tokens += operation_token

            output_json_data[utt_id]["hyp"]["hyp_{}".format(candidate_index+1)] = {}
            output_json_data[utt_id]["hyp"]["hyp_{}".format(candidate_index+1)]["text"] = hyp
            output_json_data[utt_id]["hyp"]["hyp_{}".format(candidate_index+1)]["alignment"] \
                = [ref_tokens, hyp_tokens, operation_tokens]
        
    with open(output_path, "w", encoding="utf8") as out_file:
        json.dump(output_json_data, out_file, ensure_ascii=False, indent=4)
