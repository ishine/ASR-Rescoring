import json
from typing import Dict

def get_output_format(path, max_utt, n_best):
    origin_format = json.load(open(path, "r", encoding="utf-8"))
    output_format = {}
    for utt_count, (utt_id, hyps) in enumerate(origin_format.items()):
        if utt_count == max_utt: break
        output_format[utt_id] = {}
        
        if isinstance(hyps, Dict):
            for hyp_count, (hyp_id, _) in enumerate(hyps.items()):
                if hyp_count == n_best: break
                output_format[utt_id][hyp_id] = 0
    
    return output_format 