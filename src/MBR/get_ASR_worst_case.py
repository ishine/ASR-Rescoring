# 計算 ASR 經 rescoring 最好(wer最低)與最差(wer最高)的情況

import csv
from jiwer import cer
from rescoring import parse_input_csv_row
from tqdm import tqdm

root_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/data"
train_path = root_path  + "/AISHELL1_train.csv"
dev_path = root_path  + "/AISHELL1_dev.csv"
test_path = root_path  + "/AISHELL1_test.csv"
input_data_sets = [train_path, dev_path, test_path]

output_path = root_path + "/beam_search_worst_case.txt"

n_best = 10

with open(output_path, "w") as output_file:
    for data_path in input_data_sets:
        with open(data_path) as input_csv_file:

            csv_rows = csv.reader(input_csv_file)
            header = next(csv_rows)

            origin_accumulate_wer = 0
            best_accumulate_wer = 0
            worst_accumulate_wer = 0

            count_input_num = 0
            for data in tqdm(csv_rows):
                reference, origin_best_candiddate, candidates = parse_input_csv_row(data)
                
                origin_accumulate_wer += cer(reference, origin_best_candiddate)
                
                score_list = []
                for candidate_index, candidate in enumerate(candidates):
                    score_list.append(cer(reference, candidate))

                best_accumulate_wer += min(score_list)
                worst_accumulate_wer += max(score_list)
                
                count_input_num += 1

            output_file.write(
                "file: " + data_path + "\n" +
                "ASR origin wer (no rescoring): " + str(round((origin_accumulate_wer / count_input_num)*100, 1)) + "%\n" +
                "rescoring upper bound (best case): " + str(round((best_accumulate_wer / count_input_num)*100, 1)) + "%\n" + 
                "rescoring lower bound (worst case): " + str(round((worst_accumulate_wer / count_input_num)*100, 1)) + "%\n\n")