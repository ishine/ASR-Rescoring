import sys
sys.path.append("..")
import json
import logging
from functools import partial
from typing import Dict, List

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import RescoreBert
from preprocess import get_feature
from util.arg_parser import ArgParser
from util.saving import model_saving, json_saving
from util.get_output_format import get_output_format

class MyDataset(Dataset):
    def __init__(self, data_set: List):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]


def collate(batch, config):
    utt_id = []
    hyp_id = []
    input_ids = []
    attention_masks = []
    mlm_pll_score = []
    hyps_am_score = []
    hyps_cer = []

    for data in batch:
        utt_id.append(data["utt_id"])
        hyp_id.append(data["hyp_id"])
        input_ids.append(
            torch.tensor(data["hyps_token_ids"], dtype=torch.long)
        )
        attention_masks.append(
            torch.tensor(data["attention_masks"], dtype=torch.long)
        )
        if config.task == "training":
            mlm_pll_score.append(data["mlm_pll_score"])
            if config.method in ["MD_MWER","MD_MWED"]:
                hyps_am_score.append(data["hyps_am_score"])
                hyps_cer.append(data["hyps_cer"])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    mlm_pll_score = torch.tensor(mlm_pll_score, dtype=torch.float)
    hyps_am_score = torch.tensor(hyps_am_score, dtype=torch.float)
    hyps_cer = torch.tensor(hyps_cer, dtype=torch.float)

    return{
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "mlm_pll_score": mlm_pll_score,
        "hyps_am_score": hyps_am_score,
        "hyps_cer": hyps_cer,
        "utt_id": utt_id,
        "hyp_id": hyp_id
    }

def set_dataloader(config, dataset, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=partial(collate, config=config),
        batch_size=config.batch_size * config.n_best,
        num_workers=config.dataloader.num_worker,
        shuffle=shuffle
    )
    return dataloader


def run_one_epoch(config, model, dataloader, output_score=None, grad_update=True, do_scoring=False):
    if grad_update:
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
    else:
        model.eval()

    epoch_loss = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, batch in loop:
        batch["input_ids"] = batch["input_ids"].to(config.device)
        batch["attention_masks"] = batch["attention_masks"].to(config.device)
        batch["mlm_pll_score"] = batch["mlm_pll_score"].to(config.device)

        with torch.set_grad_enabled(grad_update):
            predict_lm_score = model(
                batch["input_ids"],
                batch["attention_masks"],
            )
            if grad_update:
                batch_loss = 0

                mse_loss_fn = torch.nn.MSELoss(reduction="sum")
                MD_loss = mse_loss_fn(predict_lm_score, batch["mlm_pll_score"])

                if config.method == "MD":
                    batch_loss = MD_loss
                    batch_loss.backward()

                elif config.method == "MD_MWER":
                    batch["hyps_am_score"] = batch["hyps_am_score"].to(config.device)
                    batch["hyps_cer"] = batch["hyps_cer"].to(config.device)
                    #print( predict_lm_score, batch["hyps_am_score"])
                    mix_score = predict_lm_score + batch["hyps_am_score"]
                    #print("\n mix_score: ", mix_score)
                    mix_score = mix_score.reshape(-1, config.n_best)
                    #print("\n mix_score(reshape): ", mix_score)
                    probility = torch.softmax(-1*mix_score, dim=-1)
                    #print("\nprobility: ", probility)
                    batch["hyps_cer"] = batch["hyps_cer"].reshape(-1, config.n_best)
                    #print("\ncer: ", batch["hyps_cer"])
                    average_cer = torch.sum(batch["hyps_cer"], dim=-1) / config.n_best
                    average_cer = average_cer.unsqueeze(dim=-1)
                    #print("\n avergae: ", average_cer)
                    #print("\nbatch[\"hyps_cer\"] - average_cer", batch["hyps_cer"] - average_cer)
                    MWER_loss = torch.sum(torch.mul(probility, (batch["hyps_cer"] - average_cer)))
                    batch_loss = MWER_loss + config.md_loss_weight * MD_loss
                    batch_loss.backward()

                elif config.method == "MD_MWED":
                    batch["hyps_am_score"] = batch["hyps_am_score"].to(config.device)
                    batch["hyps_cer"] = batch["hyps_cer"].to(config.device)
                    mix_score = predict_lm_score + batch["hyps_am_score"]

                    batch["hyps_cer"] = batch["hyps_cer"].reshape(-1, config.n_best)
                    error_distribution = torch.softmax(batch["hyps_cer"], dim=1)

                    temperature = torch.sum(mix_score, dim=-1) / torch.sum(batch["hyps_cer"],dim=-1)
                    temperature = temperature.unsqueeze(dim=-1)

                    score_distribution = torch.softmax(mix_score/temperature, dim=-1)

                    MWED_loss = torch.nn.functional.kl_div(
                        torch.log(score_distribution),
                        error_distribution,
                        reduction="sum"
                    )
                    batch_loss = MWED_loss + config.md_loss_weight * MD_loss
                    batch_loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

        if do_scoring:
            for utt_id, hyp_id, lm_score in zip(batch["utt_id"], batch["hyp_id"], predict_lm_score):
            #mix_score = predict_lm_score + batch["hyps_am_score"].to("cpu")
                output_score[utt_id][hyp_id] = lm_score.item()
        else:
            epoch_loss += batch_loss.item()

    if do_scoring:
        return output_score        
    else:
        return epoch_loss / len(dataloader)


def train(config):
    train_features = get_feature(
        config,
        data_paths=config.train_feature_path,
        require_features=config.train_feature
    )
    train_set = MyDataset(train_features)
    train_loader = set_dataloader(config, train_set, shuffle=False)

    dev_features = get_feature(
        config,
        data_paths=config.dev_feature_path,
        require_features=config.dev_feature
    )
    dev_set = MyDataset(dev_features)
    dev_loader = set_dataloader(config, dev_set, shuffle=False)

    model = RescoreBert(config.model.bert)
    resume = False
    if config.resume.start_from != None and config.resume.checkpoint_path != None:
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint)
        resume = True
    model = model.to(config.device)

    train_loss_record = [0]*config.epoch
    dev_loss_record = [0]*config.epoch
    
    for epoch_id in range(config.resume.start_from if resume else 1, config.epoch+1):
        print("Epoch {}/{}".format(epoch_id, config.epoch))
        
        train_loss_record[epoch_id-1] = run_one_epoch(
            config=config,
            model=model,
            dataloader=train_loader,
            output_score=None,
            grad_update=True,
            do_scoring=False
        )
        print("epoch ", epoch_id, " train loss: ", train_loss_record[epoch_id-1])

        dev_loss_record[epoch_id-1] = run_one_epoch(
            config=config,
            model=model,
            dataloader=dev_loader,
            output_score=None,
            grad_update=True,
            do_scoring=False
        )
        print("epoch ", epoch_id, " dev loss: ", dev_loss_record[epoch_id-1], "\n")

        model_saving(config.output_path, model.state_dict(), epoch_id)
        json_saving(
            config.output_path + "/loss.json",
            {"train": train_loss_record, "dev": dev_loss_record}
        )


def score(config):
    dev_features = get_feature(
        config,
        data_paths=config.dev_feature_path,
        require_features=config.dev_feature
    )
    dev_set = MyDataset(dev_features)
    dev_loader = set_dataloader(config, dev_set, shuffle=False)

    test_features = get_feature(
        config,
        data_paths=config.test_feature_path,
        require_features=config.test_feature
    )
    test_set = MyDataset(test_features)
    test_loader = set_dataloader(config, test_set, shuffle=False)

    model = RescoreBert(config.model.bert)
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    
    dev_output_score = get_output_format(
        config.dev_output_format,
        config.max_utt,
        config.n_best
    )

    dev_output_score = run_one_epoch(
        config=config,
        model=model,
        dataloader=dev_loader,
        output_score=dev_output_score,
        grad_update=False,
        do_scoring=True
    )
    json_saving(config.output_path + "/dev_lm.json", dev_output_score)

    test_output_score = get_output_format(
        config.test_output_format,
        config.max_utt,
        config.n_best
    )

    test_output_score = run_one_epoch(
        config=config,
        model=model,
        dataloader=test_loader,
        output_score=test_output_score,
        grad_update=False,
        do_scoring=True
    )
    json_saving(config.output_path + "/test_lm.json", test_output_score)
    return


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    if config.seed != None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    if config.task == "training":
        train(config)
    elif config.task == "scoring":
        score(config)