import sys
sys.path.append("..")
import json
import logging
from typing import Dict, List

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm

from util.arg_parser import ArgParser
from util.saving import model_saving, json_saving


class MyDataset(Dataset):
    def __init__(self, data_sets: List[Dict]):
        self.data_sets = data_sets

    def __len__(self):
        return len(self.data_sets[0])
    
    def __getitem__(self, idx):
        return [data_set[idx] for data_set in self.data_sets]


def collate(batch):

    for data in batch:
        print(data)

    return 


def set_dataloader(config, dataset, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate,
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
    for _, data in loop:

        with torch.set_grad_enabled(grad_update):
            output = model(
                return_dict=True
            )

            if grad_update:
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if do_scoring:
                pass

        epoch_loss += output.loss.item()

    if not do_scoring:
        return epoch_loss / len(dataloader)
    else:
        return output_score


def train(config):
    train_hyps_token_ids = json.load(
        open(config.train_data_path + "/hyps_token_ids.json", "r", encoding="utf-8")
    )
    train_mlm_pll_score = json.load(
        open(config.train_data_path + "/mlm_pll_score.json", "r", encoding="utf-8")
    )

    train_set = MyDataset([
        train_hyps_token_ids,
        train_mlm_pll_score
    ])

    train_loader = set_dataloader(config.dataloader, train_set, shuffle=False)
    dev_loader = set_dataloader(config.dataloader, dev_loader, shuffle=False)


    if config.method == "MD_MWED":
        pass

    elif config.method == "MD_MWER":
        pass


    model = BertModel.from_pretrained(config.model.bert)
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