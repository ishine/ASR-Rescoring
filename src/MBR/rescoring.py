import csv
import math
from multiprocessing import Value
import numpy as np
from transformers.utils.dummy_sentencepiece_objects import BertGenerationTokenizer
import torch
import sys
from Bert_score.bert_score.scorer import BERTScorer
from tqdm import tqdm
from jiwer import cer
from transformers import AutoModel
from sklearn.preprocessing import normalize

root_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/data"
train_path = root_path  + "/AISHELL1_train.csv"
dev_path = root_path  + "/AISHELL1_dev.csv"
test_path = root_path  + "/AISHELL1_test.csv"

output_path = "/home/chkuo/chkuo/experiment/bertscore_MBR"
train_output_path = output_path + "/rescoring_result/train_result.txt"
dev_output_path = output_path + "/rescoring_result/dev_result.txt"
test_output_path = output_path + "/rescoring_result/test_result.txt"

output_file = open(test_output_path, "w")

model_path = "/home/chkuo/chkuo/experiment/bertscore_MBR/fine_tune_bert_weight_2"
model = AutoModel.from_pretrained(model_path)
model.eval()

n_best = 10

def parse_input_csv_row(data):
    reference = data[0]
    origin_best_candidate = data[1]
    candidates = dict()
    asr_score = dict()

    candidates = data[1:n_best + 1]
    while candidates.__contains__(""):
        candidates.remove("")

    tmp = data[n_best+1:2*n_best+1]
    while tmp.__contains__(""):
        tmp.remove("")
    scores = [float(score) for score in tmp]
    scores = normalization(scores)
    for candidate_index, score in enumerate(scores):
        asr_score[candidates[candidate_index]] = float(score)

    return reference, origin_best_candidate, candidates, asr_score

def minimum_bayesian_risk(candidates):
    candidates_mbr_score = dict()
    num_of_candidates = len(candidates)
    
    tmp_references = []
    tmp_candidates = []
    # 每次選擇一個 candidate sentence 當作 reference，去和其他 candidates 計算 recall 
    for _ in range(num_of_candidates):
        tmp_reference = candidates.pop(0)
        tmp_references += [tmp_reference] * (num_of_candidates-1)
        tmp_candidates += candidates
        candidates += [tmp_reference]
    
    precision_list, recall_list, F1_list = scorer.score(tmp_candidates, tmp_references)
    scores = [(sum(recall_list[index:index+num_of_candidates-1])) for index in range(0, len(recall_list), num_of_candidates-1)]
    scores = [score.item() for score in scores]
    scores = normalization(scores)
    for candidate_index, score in enumerate(scores):
        candidates_mbr_score[candidates[candidate_index]] = score

    return candidates_mbr_score

def write_output(reference_list, origin_best_candiddate_list, rescored_best_candidate_list):
    for index in range(len(reference_list)):
        output_file.write(
            "reference sentence: " + reference_list[index] + "\n"
            "origin best candiddate: " + origin_best_candiddate_list[index] + "\n"
            "rescored best candidate: " + rescored_best_candidate_list[index] + "\n" + "=" *50)

def rescore_with_asr_score(asr_score, candidates_rescored_score, alpha=1):
    for index in range(len(candidates_rescored_score)):
        candidates_rescored_score[index] = (1-alpha) * asr_score[index] \
                                            + alpha * candidates_rescored_score[index]
    return candidates_rescored_score

def get_best_candidate(all_candidates_mbr_score):
    mbr_best_candidate_list = []
    for data_index in range(len(all_candidates_mbr_score)):
        mbr_best_candidate = max(all_candidates_mbr_score[data_index], key=all_candidates_mbr_score[data_index].get)
        mbr_best_candidate_list.append(mbr_best_candidate)
    return mbr_best_candidate_list

def find_best_alpha(all_reference, all_asr_score, all_candidates_mbr_score):
    best_alpha = 0
    best_wer = sys.float_info.max
    
    for alpha in np.arange(0.0, 1.0, 0.01):
        all_asr_and_mbr_score = score_merge(all_asr_score, all_candidates_mbr_score, alpha=alpha)
        wer = 0
        mbr_best_candidate_list = get_best_candidate(all_asr_and_mbr_score)
        wer = cer(all_reference, mbr_best_candidate_list)
        
        if wer < best_wer:
            best_wer = wer
            best_alpha = alpha
    
    return best_alpha

def score_merge(all_asr_score, all_candidates_mbr_score, alpha = None):
    if alpha == None:
        alpha = 0.5
    all_asr_and_mbr_score = []
    for data_index in range(len(all_candidates_mbr_score)):
        tmp = dict().fromkeys(all_asr_score[data_index], 0)
        for candidate, score in all_asr_score[data_index].items():
            tmp[candidate] += (1-alpha) * score
        for candidate, score in all_candidates_mbr_score[data_index].items():
            tmp[candidate] += alpha * score
        all_asr_and_mbr_score.append(tmp)
    return all_asr_and_mbr_score

def do_train(file_path):
    with open(file_path) as input_csv_file:

        csv_rows = csv.reader(input_csv_file)
        header = next(csv_rows)

        accumulate_wer = 0
        origin_accumulate_wer = 0
        count_input_num = 1

        all_reference = []
        all_asr_score = []
        all_candidates_mbr_score = []
        
        for data in tqdm(csv_rows):
            reference, origin_best_candidate, candidates, asr_score = parse_input_csv_row(data)
            all_reference.append(reference)
            all_asr_score.append(asr_score)
            all_candidates_mbr_score.append(minimum_bayesian_risk(candidates))

        alpha = find_best_alpha(all_reference, all_asr_score, all_candidates_mbr_score)
    
    return alpha

def do_test(file_path, best_alpha):
    with open(file_path) as input_csv_file:

        csv_rows = csv.reader(input_csv_file)
        header = next(csv_rows)

        accumulate_wer = 0
        origin_accumulate_wer = 0
        count_input_num = 1

        all_reference = []
        all_origin_best_candidate = []
        all_asr_score = []
        all_candidates_mbr_score = []
        
        for data in tqdm(csv_rows):
            reference, origin_best_candidate, candidates, asr_score = parse_input_csv_row(data)

            all_reference.append(reference)
            all_origin_best_candidate.append(origin_best_candidate)
            all_asr_score.append(asr_score)
            all_candidates_mbr_score.append(minimum_bayesian_risk(candidates))

        all_asr_and_mbr_score = score_merge(all_asr_score, all_candidates_mbr_score, alpha=best_alpha)
        rescored_best_candidate_list = get_best_candidate(all_asr_and_mbr_score)

    return all_reference, all_origin_best_candidate, rescored_best_candidate_list

def normalization(input_list):
    
    v = np.array(input_list)
    normalized_v = normalize(v[:,np.newaxis], norm = "max", axis=0).ravel()
    
    # normalized_v = [math.log(v+1) for v in input_list]
    return normalized_v

if __name__ == '__main__':

    scorer = BERTScorer(lang="zh", rescale_with_baseline=True , model=model)

    alpha = do_train(test_path)
    print("alpha: ", alpha)
    all_reference, all_origin_best_candidate, rescored_best_candidate_list = do_test(test_path, alpha)

    write_output(all_reference, all_origin_best_candidate, rescored_best_candidate_list)
    
    rescored_wer = cer(all_reference, rescored_best_candidate_list)
    origin_wer = cer(all_reference, all_origin_best_candidate)

    output_file.write("average WER: " + str(rescored_wer))
    print("rescored average WER: " + str(rescored_wer))
    print("origin average WER: " + str(origin_wer))