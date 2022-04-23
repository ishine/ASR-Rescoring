import sys
sys.path.append("..")
import json
import logging
from typing import List

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM
from tqdm import tqdm

from util.arg_parser import ArgParser
from util.saving import model_saving, json_saving


class MyDataset(Dataset):
    def __init__(self, **features):
        self.data_set = []
        for feature in features:
            for f in feature:
                
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]
        

def collate(batch):
    input_ids = []
    attention_mask = []
    labels = []
    utt_id = []
    hyp_id = []
    mask_pos = []

    for data in batch:
        input_ids.append(
            torch.tensor(data["input_ids"], dtype=torch.long)
        )
        attention_mask.append(
            torch.tensor(data["attention_masks"], dtype=torch.long)
        )
        labels.append(
            torch.tensor(data["labels"], dtype=torch.long)
        )      
        utt_id.append(data["utt_id"])
        hyp_id.append(data["hyp_id"])
        mask_pos.append(data["mask_pos"])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return input_ids, attention_mask, labels, utt_id, hyp_id, mask_pos


def set_dataloader(config, dataset, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        shuffle=shuffle
    )
    return dataloader


def run_one_epoch(config, model, dataloader, output_score=None, train_mode=True, do_scoring=False):
    if train_mode:
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
    else:
        model.eval()

    epoch_loss = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, (input_ids, attention_mask, labels, utt_id, hyp_id, mask_pos) in loop:
        input_ids = input_ids.to(config.device)
        attention_masks = attention_mask.to(config.device)
        labels = labels.to(config.device)

        with torch.set_grad_enabled(train_mode):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels,
                return_dict=True
            )

            if train_mode:
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if do_scoring:
                token_logits = output.logits[range(len(output.logits)), mask_pos, :]
                token_score = token_logits.log_softmax(dim=-1)
                #print(token_logits.softmax(dim=-1))
                masked_token_ids = labels[range(len(labels)), mask_pos]
                token_score = token_score[range(len(token_score)), masked_token_ids].tolist()
                for u_id, h_id, s in zip(utt_id, hyp_id, token_score):
                    output_score[u_id][h_id] += s

        epoch_loss += output.loss.item()

    if not do_scoring:
        return epoch_loss / len(dataloader)
    else:
        return output_score


def train(config):
    train_hyps_token_ids = json.load(
        open(config.train_data_path + "/hyps_token_ids.json", "r", encoding="utf-8")
    )[:config.num_of_data]

    train_mlm_pll_score = json.load(
        open(config.train_data_path + "/mlm_pll_score.json", "r", encoding="utf-8")
    )[:config.num_of_data]

    train_set = MyDataset()

    if config.method == "MD_MWED":
        train_set = MyDataset(json.load(
            open(config.train_data_path, "r", encoding="utf-8")
        ))[:config.num_of_data]
        dev_set = MyDataset(json.load(
            open(config.dev_data_path, "r", encoding="utf-8")
        ))[:config.num_of_data]

    elif config.method == "MD_MWER":
        train_set = MyDataset(json.load(
            open(config.train_data_path, "r", encoding="utf-8")
        ))[:config.num_of_data]
        dev_set = MyDataset(json.load(
            open(config.dev_data_path, "r", encoding="utf-8")
        ))[:config.num_of_data]
        
    return


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