import bert_score

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