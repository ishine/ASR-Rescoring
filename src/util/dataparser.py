import json
from types import SimpleNamespace
from weakref import ref
from util.config import parse_config

class DataParser():
    def __init__(self, json_file_path: str, max_utts: int = -1, n_best: int = -1):
        self.json_file_path =json_file_path
        self.max_utts = max_utts
        self.n_best = n_best

    def parse(self):
        json_data = json.load(
            open(self.json_file_path, "r", encoding="utf-8")
        )       

        self.data = []
        for utt_count, (utt_id, utt_content) in enumerate(json_data.items(), 0):
            if utt_count == self.max_utts:
                break
            
            ref = SimpleNamespace()
            setattr(ref, "text", utt_content["ref"])
            
            hyps = []
            for hyp_count, (hyp_id, hyp_content) in enumerate(utt_content["hyp"].items(), 0):
                if hyp_count == self.n_best:
                    break
                hyp = SimpleNamespace()
                setattr(hyp, "id", hyp_id)
                setattr(hyp, "text", hyp_content["text"])
                setattr(hyp, "score", hyp_content["score"])
                setattr(hyp, "cer", hyp_content["cer"])
                hyps.append(hyp)

            utt = SimpleNamespace()
            setattr(utt, "id", utt_id)
            setattr(utt, "ref", ref)
            setattr(utt, "hyps", hyps)            
            
            self.data.append(utt)

        return self.data