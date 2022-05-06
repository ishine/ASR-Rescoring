from jiwer import cer
import json
ref_file = "../espnet_data/alfred/dev/ref_text.json"
pred_file = "result/3_best_align_not_fuse/lr_10-6/dev_pred.json"

ref_json = json.load(open(ref_file, "r", encoding="utf-8"))
pred_json = json.load(open(pred_file, "r", encoding="utf-8"))

print(cer(list(ref_json.values()), list(pred_json.values())))