import sys
from typing import Dict, List
sys.path.append("..")
import json
from tqdm import tqdm

def get_feature(config, data_paths, require_features):

    feature_set = {}
    for path, feature in zip(data_paths, require_features):
        feature_set[feature] = json.load(open(path, "r", encoding="utf-8"))

    # initialize the output data format
    refs, hyps = [], []
    for feature, feature_json in feature_set.items():
        
        for utt_count, utt_content in enumerate(feature_json.values()):
            if utt_count == config.max_utt: break
            if feature == "ref_text":
                refs.append(utt_content)
                continue

            utt_hyps = []
            for hyp_count, hyp_content in enumerate(utt_content.values()):
                if hyp_count == config.n_best: break
                elif feature == "hyps_text":
                    utt_hyps.append(hyp_content)
            hyps.append(utt_hyps)
            
    return refs, hyps