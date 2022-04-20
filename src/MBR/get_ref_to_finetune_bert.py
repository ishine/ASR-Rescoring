import pandas as pd

data_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/data/AISHELL1_test.csv"

df = pd.read_csv(data_path)

output_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/data/AISHELL1_test_ref.txt"
output_file = open(output_path, "w")

for ref in list(df.iloc[:,0]):
  output_file.write(ref + "\n")