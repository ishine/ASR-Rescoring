import sys
sys.path.append("/home/chkuo/chkuo/experiment/ASR-Rescoring/src")
#sys.path.append("D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\src")
from util.arg_parser import ArgParser 
from trainer import BERTsemTrainer, BERTalsemTrainer
from scorer import BERTsemScorer
from rescorer import Rescorer

if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    if config.action == "train":
        if config.method == "bertsem":
            trainer = BERTsemTrainer(config.setting)
        if config.method == "bertalsem":
            trainer = BERTalsemTrainer(config.setting)
        trainer.train()
    
    elif config.action == "score":
        if config.method == "bertsem":
            scorer = BERTsemScorer(config.setting)
        scorer.score()

    elif config.action == "rescore":
        rescorer = Rescorer(config.setting)
        rescorer.rescore()
        