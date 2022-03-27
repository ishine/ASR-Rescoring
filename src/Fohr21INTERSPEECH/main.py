import sys
#sys.path.append("/home/chkuo/chkuo/experiment/ASR-Rescoring/src")
sys.path.append("D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\src")
from util.arg_parser import ArgParser 
from trainer import BERTsemTrainer, BERTalsemTrainer
from scorer import BERTsemScorer

if __name__ == "__main__":
    arg_parser = ArgParser()
    action, method, setting = arg_parser.parse()

    if action == "train":
        if method == "bertsem":
            trainer = BERTsemTrainer(setting)
        if method == "bertalsem":
            trainer = BERTalsemTrainer(setting)
        trainer.train()
    
    elif action == "score":
        if method == "bertsem":
            scorer = BERTsemScorer(setting)
        scorer.score()