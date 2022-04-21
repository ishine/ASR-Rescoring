import sys
sys.path.append("..")
import json
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
    def __init__(self, data_set: List):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]
        

def collate(batch):
    input_ids = []
    attention_mask = []
    labels = []

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
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return input_ids, attention_mask, labels

def set_dataloader(config, dataset, for_train=False):
    if for_train:
        shuffle=config.shuffle
    else:
        shuffle=False

    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        shuffle=shuffle
    )
    return dataloader

def run_one_epoch(config, model, dataloader, train_mode: bool):
    if train_mode:
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
    else:
        model.eval()

    epoch_loss = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, (input_ids, attention_mask, labels) in loop:
        input_ids =input_ids.to(config.device)
        attention_masks = attention_mask.to(config.device)
        labels = attention_mask.to(config.device)

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

        epoch_loss += output.loss.item()

    return epoch_loss / len(dataloader)

def mlm_finetune_bert(config):
    train_set = MyDataset(json.load(
        open(config.train_data_path, "r", encoding="utf-8")
    ))
    dev_set = MyDataset(json.load(
        open(config.dev_data_path, "r", encoding="utf-8")
    ))

    train_loader = set_dataloader(config.dataloader, train_set, True)
    dev_loader = set_dataloader(config.dataloader, dev_set, False)

    model = BertForMaskedLM.from_pretrained(config.model.bert)
    model = model.to(config.device)

    train_loss_record = [0]*config.epoch
    dev_loss_record = [0]*config.epoch

    for epoch_id in range(1, config.epoch+1):
        print("Epoch {}/{}".format(epoch_id, config.epoch))
        
        train_loss_record[epoch_id-1] = run_one_epoch(
            config,
            model,
            train_loader,
            train_mode=True
        )
        print("epoch ", epoch_id, " train loss: ", train_loss_record[epoch_id-1])

        dev_loss_record[epoch_id-1] = run_one_epoch(
            config,
            model,
            dev_loader,
            train_mode=False
        )
        print("epoch ", epoch_id, " dev loss: ", dev_loss_record[epoch_id-1], "\n")

        model_saving(config.output_path, model.state_dict(), epoch_id)
        json_saving(
            config.output_path + "/loss.json",
            {"train": train_loss_record, "dev": dev_loss_record}
        )


def pll_bert_scoring(config):
    return


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    if config.seed != None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    if config.task == "training":
        mlm_finetune_bert(config)
    elif config.task == "scoring":
        pll_bert_scoring(config)