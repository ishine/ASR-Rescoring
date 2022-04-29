import sys
import argparse
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--cer", type=str, required=True, 
        help="input cer json file")
    parser.add_argument("--n_best", type=str, required=True, 
        help="n best")
    args = parser.parse_args()

    # read cer and other basic informations   
    cer_json = json.load(
        open(args.cer, "r", encoding="utf-8")
    )
    n_best = int(args.n_best)
    utt_num = len(cer_json)

    # compute oracle distribution
    oracle_distribution = {pos: 0 for pos in range(n_best)}
    for hyps in cer_json.values():
        max_cer = sys.maxsize
        oracle_pos = -1
        for pos, cer in zip(range(n_best), hyps.values()):
            if cer < max_cer:
                max_cer = cer
                oracle_pos = pos
        oracle_distribution[oracle_pos] += 1

    # show result
    print("oracle distribution: {pos in n-best: count} => ", oracle_distribution)
    plot = plt.bar(
        oracle_distribution.keys(),
        oracle_distribution.values(),
    )
    plt.xlabel("oracle position in n-best")
    plt.ylabel("oracle count")
    plt.show()