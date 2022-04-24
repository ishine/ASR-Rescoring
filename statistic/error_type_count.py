import json
import argparse
from tqdm import tqdm

def error_type_statistic(json_data):

    output = {}
    output["Unchange token number"] = 0
    output["Substitution token number"] = 0
    output["Insertion token number"] = 0
    output["Deletion token number"] = 0

    for utt in tqdm(json_data.values(), total=len(json_data.values())):
        for hyp in utt["hyp"].values():
            operation_tokens = hyp["alignment"][2]
            for token in operation_tokens:
                if token == "U":
                    output["Unchange token number"] += 1
                elif token == "S":
                    output["Substitution token number"] += 1
                elif token == "I":
                    output["Insertion token number"] += 1
                elif token == "D":
                    output["Deletion token number"] += 1
    
    return output


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, 
        help="statistic type")
    parser.add_argument("--in_file", type=str, required=True, 
        help="input json file")
    parser.add_argument("--out_file", type=str, required=True, 
        help="output statistic result file")

    args = parser.parse_args()
    
    
    with open(args.in_file, "r") as in_file:
        json_data = json.load(in_file)
    
    
    if args.type == "error_type":
        result = error_type_statistic(json_data)

    
    with open(args.out_file, "w") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=4)
