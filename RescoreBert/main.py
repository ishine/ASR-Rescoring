import sys

from matplotlib.pyplot import get
sys.path.append("..")
import json
import logging
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


class MyDataset(Dataset):
    def __init__(self, data_set: List):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]


def collate(batch, config):
    input_ids = []
    attention_masks = []
    mlm_pll_score = []
    utt_id = []
    hyp_id = []

    for data in batch:
        input_ids.append(
            torch.tensor(data["hyps_token_ids"], dtype=torch.long)
        )
        attention_masks.append(
            torch.tensor(data["attention_masks"], dtype=torch.long)
        )
        mlm_pll_score.append(
            torch.tensor(data["mlm_pll_score"], dtype=torch.long)
        )      
        utt_id.append(data["utt_id"])
        hyp_id.append(data["hyp_id"])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    if config.method == "MD":
        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "mlm_pll_score": mlm_pll_score,
            "utt_id": utt_id,
            "hyp_id": hyp_id
        }


def set_dataloader(config, dataset, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate(config=config),
        batch_size=config.batch_size,
        num_workers=config.num_worker,
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

        with torch.set_grad_enabled(grad_update):
            predict_lm_score = model(
                batch["input_ids"].to(config.device),
                batch["attention_masks"].to(config.device),
                return_dict=True
            )

            if grad_update:
                batch_loss = 0

                mse_loss_fn = torch.nn.MSELoss(reduction="sum")
                MD_loss = mse_loss_fn(predict_lm_score, batch["mlm_pll_score"])

                if config.method == "MD":
                    batch_loss = MD_loss
                    batch_loss.backward()

                elif config.method == "MD_MWER":
                    mix_score = predict_lm_score + batch["hyps_am_score"]
                    probility = torch.softmax(-1*mix_score, dim=1)
                    average_cer = torch.sum(batch["hyps_cer"], dim=1) / config.n_best
                    average_cer = average_cer.unsqueeze(dim=1)
                    MWER_loss = torch.sum(torch.mul(probility, (batch["hyps_cer"] - average_cer)))
                    batch_loss = MWER_loss + config.md_weight * MD_loss
                    batch_loss.backward()

                elif config.method == "MD_MWED":
                    mix_score = predict_lm_score + batch["hyps_am_score"]

                    batch_loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

        if do_scoring:
            pass

        epoch_loss += batch_loss

    if not do_scoring:
        return epoch_loss / len(dataloader)
    else:
        return output_score


def train(config):
    train_features = get_feature(
        config,
        data_paths=config.train_feature_path,
        require_features=config.train_feature
    )[:config.num_of_data]
    train_set = MyDataset(train_features)
    train_loader = set_dataloader(config.dataloader, train_set, shuffle=False)

    dev_features = get_feature(
        config,
        data_paths=config.dev_feature_path,
        require_features=config.dev_feature
    )[:config.num_of_data]
    dev_set = MyDataset(dev_features)
    dev_loader = set_dataloader(config.dataloader, dev_set, shuffle=False)

    model = RescoreBert(config.model.bert)
    model = model.to(config.device)

    train_loss_record = [0]*config.epoch
    dev_loss_record = [0]*config.epoch
    
    for epoch_id in range(1, config.epoch+1):
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