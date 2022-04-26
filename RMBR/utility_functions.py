import bert_score
from jiwer import cer 
class BaseFunction():
    def __init__(self, config) -> None:
        self.config = config
    def score(self):
        raise NotImplementedError

class BertScoreFunction(BaseFunction):
    def __init__(self, config) -> None:
        super().__init__(config)

    def score(self, cands, refs):
        _, recall, __ = bert_score.score(
            cands,
            refs,
            lang=self.config.language,
            verbose=True,
            device=self.config.device,
            batch_size=self.config.batch_size
        )
        return recall

class CerScoreFunction(BaseFunction):
    def __init__(self, config) -> None:
        super().__init__(config)

    def score(self, cands, refs):
        similarity_scores = []
        for cand, ref in zip(cands, refs):
            error_rate = cer(ref, cand)
            similarity_scores.append(1 - error_rate)
        return similarity_scores