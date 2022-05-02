from jiwer import cer
import json
ref_file = "/home/chkuo/chkuo/experiment/ASR-Rescoring/espnet_data/alfred/test/ref_text.json"
pred_file = "/home/chkuo/chkuo/experiment/ASR-Rescoring/CorrectBart/result/one_hyp/test_pred.json"

ref_json = json.load(open(ref_file, "r", encoding="utf-8"))
pred_json = json.load(open(pred_file, "r", encoding="utf-8"))

print(cer(list(ref_json.values()), list(pred_json.values())))