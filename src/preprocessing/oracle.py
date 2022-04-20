import argparse
import json
import matplotlib.pyplot as plt
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
    
    n_best = len(hypotheses[0])
    utt_num = len(hypotheses)

    oracle_count = {pos: 0 for pos in range(n_best)}
    hyp = []
    for utt_hypotheses, utt_hyp_cers in zip(hypotheses, hypotheses_cers):
        min_value = min(utt_hyp_cers)
        min_value_pos = utt_hyp_cers.index(min_value)
        oracle_count[min_value_pos] += 1
        min_id = utt_hyp_cers.index(min_value)
        hyp.append(utt_hypotheses[min_id])
    

    print(cer(ref, hyp))

    print(oracle_count)
    oracle_dictribution = [num/utt_num for num in oracle_count.values()]
    print(oracle_dictribution)
    plt.hist(
        oracle_count.values(),
        bins=n_best,
        density=True,
        color = 'lightblue',
        cumulative = False,
        label = "Height"
    )
    plt.ylabel("probility")
    plt.show()