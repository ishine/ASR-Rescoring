import json
from types import SimpleNamespace
from typing import List
from collections.abc import Iterable
from util.config import parse_config

class DataParser():
    def __init__(self, json_file_path: str, max_utts: int = -1, n_best: int = -1):
        self.json_file_path = json_file_path
        self.max_utts = max_utts
        self.n_best = n_best

    def parse(self):
        json_data = json.load(
            open(self.json_file_path, "r", encoding="utf-8")
        )       

        data = Data(self.n_best)

        for utt_count, (utt_ID, utt_content) in enumerate(json_data.items(), 0):
            if utt_count == self.max_utts:
                break
            
            ref = Reference(utt_content["ref"])
            utt = Utterance(utt_ID, ref)

            for hyp_count, (hyp_ID, hyp_content) in enumerate(utt_content["hyp"].items(), 0):
                if hyp_count == self.n_best:
                    break
                
                utt.add_hypothesis(
                    Hypothesis(
                        hyp_ID,
                        hyp_content["text"], 
                        hyp_content["score"], 
                        hyp_content["cer"]
                    )
                )

            data.add_utterance(utt)
        
        return data


class Hypothesis():
    def __init__(self, ID: str, text: str, score: float, cer: float) -> None:
        self.ID = ID
        self.text = text
        self.score = score
        self.cer = cer
    
    def set_index(self, index: int):
        self.index = index 


class Reference():
    def __init__(self, text) -> None:
        self.text = text


class Utterance():
    def __init__(self, utt_ID: str, ref: Reference) -> None:
        self.ID = utt_ID
        self.ref = ref
        self.hyps = []
        self.num_hyp = 0

    def add_hypothesis(self, hyp: Hypothesis):
        hyp.set_index(self.num_hyp)
        self.num_hyp += 1
        self.hyps.append(hyp)

    def set_index(self, index: int):
        self.index = index


class Data():
    def __init__(self, n_best) -> None:
        self.utt_set = []
        self.num_utt = 0
        self.n_best = n_best

    def add_utterance(self, utt: Utterance):
        utt.set_index(self.num_utt)
        self.num_utt += 1
        self.utt_set.append(utt)

    def update_score(self, utt_ID: str, hyp_ID: str, score: int):
        for utt in self.utt_set:
            if utt.ID == utt_ID:
                for hyp in utt.hyps:
                    if hyp.ID == hyp_ID:
                        hyp.score = score

    def update_scores(self, scores: List[List]):
        for utt, row_of_scores in zip(self.utt_set, scores):
            for hyp, score in zip(utt.hyps, row_of_scores):
                hyp.score = score
    
    def size(self):
        return (self.num_utt, self.n_best)

    def to_json(self):
        json_format_data = {}
        for utt in self.utt_set:
            hyp_set = {}
            for hyp in utt.hyps:
                hyp_set[hyp.ID] = {
                    "score": hyp.score,
                    "text": hyp.text,
                    "cer": hyp.cer
                }

            json_format_data[utt.ID] = {
                "ref": utt.ref.text,
                "hyp": hyp_set
            }

        return json_format_data