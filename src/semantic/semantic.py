import sys
sys.path.append("D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\src")

from util.parse_json import parse_json


hyp_cers = parse_json(
    "D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\data\\test.am.json",
    requirements=["hyp_cer"],
    max_utts=1,
    n_best=15,
    flatten=False
)

print(hyp_cers)

counter = 0
for row in hyp_cers:
    for _ in range(len(row)):
        hyp_cer_i = row.pop(0)
        for  hyp_cer_j in row:
            if hyp_cer_i != hyp_cer_j:
                counter += 1
        print(counter)
        row.append(hyp_cer_i)
print(counter)