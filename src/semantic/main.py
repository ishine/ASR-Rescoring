import sys
sys.path.append("D:\\NTUST\\NLP\\experiments\\ASR-Rescoring\\src")
from util.arg_parser import ArgParser 
from trainer import BERTsemTrainer, BERTalsemTrainer

if __name__ == "__main__":
    arg_parser = ArgParser()
    action, method, config = arg_parser.parse()

    if action == "train":
        if method == "bertsem":
            trainer = BERTsemTrainer(config)
        if method == "bertalsem":
            trainer = BERTalsemTrainer(config)

        trainer.train()