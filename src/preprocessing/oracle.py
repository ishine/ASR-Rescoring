import argparse
import json
from jiwer import cer
from parse_json import parse_json

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, 
        help="input json file")

    args = parser.parse_args()

    ref, hypotheses, hypotheses_cers = parse_json(
        file_path=args.file, 
        requirements=["ref_text", "hyp_text", "hyp_cer"],
    )
    
    hyp = []
    for utt_hypotheses, utt_hyp_cers in zip(hypotheses, hypotheses_cers):
        min_value = min(utt_hyp_cers)
        min_id = utt_hyp_cers.index(min_value)
        hyp.append(utt_hypotheses[min_id])
    

    print(cer(ref, hyp))