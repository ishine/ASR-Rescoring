import json
from tqdm import tqdm
from jiwer import cer

from levenshtein import levenshtein_distance_alignment

# espnet asr recognition result path
input_root = "/home/chkuo/chkuo/experiment/bert_semantic_rescoring/train_sp_pytorch_train"
train_path = input_root + "/decode_train_sp_decode_lm_4/data.json"
dev_path = input_root + "/decode_dev_decode_lm_4/data.json"
test_path = input_root + "/decode_test_decode_lm_4/data.json"
inputs = [train_path, dev_path]

# json output files path 
output_root = "/home/chkuo/chkuo/experiment/ASR-Rescoring/data"
train_dataset_path = output_root + "/train.am.json"
dev_dataset_path = output_root + "/dev.am.json"
test_dataset_path = output_root + "/test.am.json"
outputs = [train_path, dev_path]

for input_path, output_path in zip(inputs, outputs):
    with open(input_path, "r") as in_file:
        input_json_data = json.load(in_file)

    output_json_data = {}

    for utt_id, utt_content in tqdm(input_json_data["utts"].items()):

        hypotheses = utt_content["output"]
        
        # 取得 reference sentence
        ref = hypotheses[0]["text"]
        
        output_json_data[utt_id] = {}
        output_json_data[utt_id]["ref"] = ref
        output_json_data[utt_id]["hyp"] = {}

        for hyp_id, hypothesis in enumerate(hypotheses, start=1):
            output_json_data[utt_id]["hyp"]["hyp_{}".format(hyp_id)] = {}

            output_json_data[utt_id]["hyp"]["hyp_{}".format(hyp_id)]["score"] = \
                hypothesis["score"]

            hyp = hypothesis["rec_text"].strip("<eos>")
            output_json_data[utt_id]["hyp"]["hyp_{}".format(hyp_id)]["text"] = \
                hyp

            output_json_data[utt_id]["hyp"]["hyp_{}".format(hyp_id)]["cer"] = \
                cer(ref, hyp)

            output_json_data[utt_id]["hyp"]["hyp_{}".format(hyp_id)]["alignment"] = \
                levenshtein_distance_alignment(reference=ref, hypothesis=hyp)
                
    with open(output_path, "w", newline = "") as out_file:
        json.dump(output_json_data, out_file, ensure_ascii=False, indent=4)
