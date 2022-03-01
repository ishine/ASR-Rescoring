import os
import argparse

import ruamel.yaml as yaml
from jiwer import cer

from train import DomainAdaptation, MLM_distill
from inference import SentencelevelScoring, TokenlevelScoring, get_recog_data
from rescorer import Rescorer
from util.config import parse_config

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

        if config.train.type == "domain_adaptation":
            DA = DomainAdaptation(config.train)
            DA.prepare_train_set()
            DA.prepare_train_loader()
            DA.train()
        elif config.train.type == "MLM_distillation":
            MD = MLM_distill(config.train)
            MD.prepare_train_set()
            MD.prepare_train_loader()
            MD.train()

    # do inference
    if "scoring" in config.actions:
        if config.scoring.type == "token_level":
            scorer = TokenlevelScoring(config.scoring)
            scorer.prepare_inference_set()
            scorer.prepare_inference_loader()
            scorer.score()
        elif config.scoring.type == "sentence_level":
            scorer = SentencelevelScoring(config.scoring)
            scorer.prepare_inference_set()
            scorer.prepare_inference_loader()
            scorer.score()

    if "rescoring" in config.actions:
        rescorer = Rescorer(config.rescoring)
        # 用dev set找最佳weight
        best_weight, best_cer = rescorer.find_best_weight()

        print("best_weight: ", best_weight)

        test_ref_text = get_recog_data(
            config.rescoring.test_asr_data_path,
            type="ref",
            max_utts=config.rescoring.max_utts)


        test_hyp_text = get_recog_data(
            config.rescoring.test_asr_data_path,
            type="hyp_text",
            max_utts=config.rescoring.max_utts)


        # rescore test set
        test_ASR_score = get_recog_data(
            config.rescoring.test_asr_data_path,
            type="hyp_score",
            max_utts=config.rescoring.max_utts)
        
        test_LM_score = get_recog_data(
            config.rescoring.test_lm_data_path,
            type="hyp_score",
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

        print("cer: ", cer)