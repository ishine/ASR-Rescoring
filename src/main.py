import os
import argparse
import ruamel.yaml as yaml
from jiwer import cer

from mlm import MaskedLanguageModelTraining, PLLScoring
from mlm_distillation import MLMDistill, MDScoring
from mwer import MWER_Training, MWED_Training, MWER_MWED_Inference
from md_mwer import MD_MWER_Training, MD_MWED_training

from error_detection_training import ErrorDetectionTraining

from rescorer import Rescorer

from util.config import parse_config
from util.parse_json import parse_json

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, 
        help="configuration file path")
    args = parser.parse_args()

    # parse configuration
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config = parse_config(config)

    # do train
    if "train" in config.actions:
        if os.path.isdir(config.train.output_path) == False:
            os.mkdir(config.train.output_path, mode=0o755)

        if config.train.type == "MLM":
            MaskedLanguageModelTraining(config.train)
        elif config.train.type == "MLM_distillation":
            MLMDistill(config.train)
        elif config.train.type == "MWER":
            MWER = MWER_Training(config.train)
            
            train_dataset = MWER.prepare_dataset(config.train.train_data_path)
            dev_dataset = MWER.prepare_dataset(config.train.dev_data_path)

            train_dataloader = MWER.prepare_dataloader(train_dataset)
            dev_dataloader = MWER.prepare_dataloader(dev_dataset)
            
            MWER.train(train_dataloader, dev_dataloader)
        elif config.train.type == "MD_MWER":
            MD_MWER_Training(config.train)
        elif config.train.type == "MD_MWED":
            MD_MWED_training(config.train)
        elif config.train.type == "error_detection_training":
            ED = ErrorDetectionTraining(config.train)
            
            train_dataset = ED.prepare_dataset(config.train.train_data_path)
            dev_dataset = ED.prepare_dataset(config.train.dev_data_path)

            train_dataloader = ED.prepare_dataloader(train_dataset)
            dev_dataloader = ED.prepare_dataloader(dev_dataset)

            ED.train(train_dataloader, dev_dataloader)
        elif config.train.type == "MWED":
            MWED = MWED_Training(config.train)
            
            train_dataset = MWED.prepare_dataset(config.train.train_data_path)
            dev_dataset = MWED.prepare_dataset(config.train.dev_data_path)

            train_dataloader = MWED.prepare_dataloader(train_dataset)
            dev_dataloader = MWED.prepare_dataloader(dev_dataset)
            
            MWED.train(train_dataloader, dev_dataloader)
    # do inference
    if "scoring" in config.actions:
        if config.scoring.type == "PLL":
            PLLScoring(config.scoring)
        elif config.scoring.type == "MLM_distillation" or \
            config.scoring.type=="MD_MWER" or \
            config.scoring.type=="MD_MWED":
            MDScoring(config.scoring)
        elif config.scoring.type == "MWER":
            scorer = MWER_MWED_Inference(config.scoring)
            dataset = scorer.prepare_dataset()
            dataloader = scorer.prepare_dataloader(dataset)
            scorer.scoring(dataloader)
        elif config.scoring.type == "error_detection_training":
            scorer = ErrorDetectionInference(config.scoring)
            dataset = scorer.prepare_dataset()
            dataloader = scorer.prepare_dataloader(dataset)
            scorer.scoring(dataloader)


    if "rescoring" in config.actions:
        rescorer = Rescorer(config.rescoring)
        # 用dev set找最佳weight
        best_weight, best_cer = rescorer.find_best_weight()

        print("best_weight: ", best_weight)

        test_ASR_score, test_ref_text, test_hyp_text = parse_json(
            config.rescoring.test_asr_data_path,
            requirements=["hyp_score", "ref_text", "hyp_text"],
            max_utts=config.rescoring.max_utts)

        test_LM_score = parse_json(
            config.rescoring.test_lm_data_path,
            requirements=["hyp_score"],
            max_utts=config.rescoring.max_utts)
        
        final_score = rescorer.rescore(
            best_weight,
            test_ASR_score,
            test_LM_score,
            test_hyp_text
        )

        # 取出最高分的hyp sentences
        predict_text = rescorer.get_highest_score_hyp(final_score, test_hyp_text)

        # 計算error rate
        cer = cer(test_ref_text, predict_text)

        print("test cer: ", cer)