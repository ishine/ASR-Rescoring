import json
from tqdm import tqdm
from jiwer import cer

from align import levenshtein_distance_alignment

# espnet asr recognition result path
'''
train_in_path = "../alfred/train_data.json"
dev_in_path = "../alfred/dev_data.json"
test_in_path = "../alfred/test_data.json"
'''
train_in_path = "../origin/espnet_train_data.json"
dev_in_path = "../origin/espnet_dev_data.json"
test_in_path = "../origin/espnet_test_data.json"

inputs = [train_in_path, dev_in_path, test_in_path]

# json output files path 
'''
train_out_path = "../alfred/train/"
dev_out_path = "../alfred/dev/"
test_out_path = "../alfred/test/"
'''
train_out_path = "../train/"
dev_out_path = "../dev/"
test_out_path = "../test/"

outputs = [train_out_path, dev_out_path, test_out_path]

for input_path, output_path in zip(inputs, outputs):
    with open(input_path, "r") as in_file:
        input_json_data = json.load(in_file)

    output_ref_text_json = {}
    output_hyps_jsons = {
        "hyps_text": {},
        "hyps_score": {},
        "hyps_cer": {},
        "hyp_alignment": {}
    }
    for utt_id, utt_content in tqdm(input_json_data["utts"].items()):

        hyps = utt_content["output"]
        ref_text = hyps[0]["text"]
        
        output_ref_text_json[utt_id] = ref_text
        for json_data in output_hyps_jsons.values():
            json_data[utt_id] = {}
        
        for hyp_id, hyp in enumerate(hyps, start=1):
            hyp_text = hyp["rec_text"].strip("<eos>")
            output_hyps_jsons["hyps_text"][utt_id][f"hyp_{hyp_id}"] = \
                hyp_text

            output_hyps_jsons["hyps_score"][utt_id][f"hyp_{hyp_id}"] = \
                hyp["score"]

            output_hyps_jsons["hyps_cer"][utt_id][f"hyp_{hyp_id}"] = \
                cer(ref_text, hyp_text)

            output_hyps_jsons["hyp_alignment"][utt_id][f"hyp_{hyp_id}"] = \
                levenshtein_distance_alignment(ref_text, hyp_text)


    with open(output_path + "ref_text.json", "w", newline = "") as out_file:
        json.dump(output_ref_text_json, out_file, ensure_ascii=False, indent=4)

    for file_name, data in output_hyps_jsons.items():
        with open(output_path + file_name + ".json", "w", newline = "") as out_file:
            json.dump(data, out_file, ensure_ascii=False, indent=4)
