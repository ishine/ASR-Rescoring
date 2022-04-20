# 統計資料集中每筆語音資料經 ASR 找出多少 candidate sentences
#
# 例如：產生 10 個candidates sentences 有多少筆
# 　　　產生 9  個candidates sentences 有多少筆
# 　　　            ...
# 　　　            ...
# 　　　產生 1  個candidates sentence  有多少筆

import pandas as pd
import numpy as np

root_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/data"
train_path = root_path  + "/AISHELL1_train.csv"
dev_path = root_path  + "/AISHELL1_dev.csv"
test_path = root_path  + "/AISHELL1_test.csv"
input_data_sets = [train_path, dev_path, test_path]

output_path = root_path + "/statistic.txt"

n_best = 10

with open(output_path, "w") as output_file:
    for data_path in input_data_sets:
        with open(data_path) as input_csv_file:

            df = pd.read_csv(input_csv_file)
            data = df.iloc[:, 1:n_best+1]

            statistic_result = data.count().values
            
            statistic_result = np.append(statistic_result, np.zeros(1))

            output_string = ""
            for index in range(len(statistic_result)-1):
                output_string += "have exactly {} candidate sentences: ".format(index+1)
                output_string += str(int(statistic_result[index] - statistic_result[index+1])) + " rows" + "\n"

            print(output_string)
            output_file.write("file: " + data_path + "\n" + output_string + "\n")