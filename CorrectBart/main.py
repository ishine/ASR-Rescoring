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

from get_feature import get_feature
from model import CorrectBart, NBestAlignCorrectBart
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
    input_ids = []
    attention_masks = []
    ref_token_ids = []

    for data in batch:
        utt_id.append(data["utt_id"])
        
        input_ids.append(
            torch.tensor(data["hyps_token_ids"], dtype=torch.long)
        )
        
        attention_masks.append(
            torch.tensor(data["attention_masks"], dtype=torch.long)
        )
        
        if config.task == "training":
            ref_token_ids.append(
                torch.tensor(data["ref_token_ids"], dtype=torch.long)
            )

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    if config.task == "training":
        ref_token_ids = pad_sequence(ref_token_ids, batch_first=True)
    
    return{
        "utt_id": utt_id,
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "ref_token_ids": ref_token_ids
    }


def set_dataloader(config, dataset, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=partial(collate, config=config),
        batch_size=config.batch_size,
        num_workers=config.dataloader.num_worker,
        shuffle=shuffle
    )
    return dataloader


def run_one_epoch(config, model, dataloader, output=None, grad_update=True, train=False):
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

        if train:
            batch["ref_token_ids"] = batch["ref_token_ids"].to(config.device)
            with torch.set_grad_enabled(grad_update):
                batch_loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_masks"],
                    labels=batch["ref_token_ids"]
                )
                epoch_loss += batch_loss.item()
            
            if grad_update:
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        else:
            predictions = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_masks"]
            )
            for utt_id, pred in zip(batch["utt_id"], predictions):
                output[utt_id]=pred

    if train:
        return epoch_loss / len(dataloader)
    else:
        return output        


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

    if config.method in ["one_hyp", "n_best_align_not_fuse"]:
        model = CorrectBart(config.model.bart)
    elif config.method == "n_best_align":
        model = NBestAlignCorrectBart(
            config.model.bart,
            config.model.alignment_embedding,
            config.n_best
        )
        
    if config.resume.start_from != None and config.resume.checkpoint_path != None:
        resume = True
        checkpoint = torch.load(config.resume.checkpoint_path)
        model.load_state_dict(checkpoint)
        loss_record = json.load(
            open(config.output_path + "/loss.json", "r", encoding="utf-8")
        )
        train_loss_record = loss_record["train"]
        dev_loss_record = loss_record["dev"]
    else:
        resume = False
        train_loss_record = []
        dev_loss_record = []
    model = model.to(config.device)
    
    for epoch_id in range(config.resume.start_from if resume else 1, config.epoch+1):
        print("Epoch {}/{}".format(epoch_id, config.epoch))
        
        train_loss = run_one_epoch(
            config=config,
            model=model,
            dataloader=train_loader,
            output=None,
            grad_update=True,
            train=True
        )
        print("epoch ", epoch_id, " train loss: ", train_loss, "\n")
        train_loss_record.append(train_loss)

        dev_loss = run_one_epoch(
            config=config,
            model=model,
            dataloader=dev_loader,
            output=None,
            grad_update=False,
            train=True
        )
        print("epoch ", epoch_id, " dev loss: ", dev_loss, "\n")
        dev_loss_record.append(dev_loss)
        
        model_saving(config.output_path, model.state_dict(), epoch_id)
        json_saving(
            config.output_path + "/loss.json",
            {"train": train_loss_record, "dev": dev_loss_record}
        )


def inference(config):
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

    if config.method in ["one_hyp", "n_best_align_not_fuse"]:
        model = CorrectBart(config.model.bart)
    elif config.method == "n_best_align":
        model = NBestAlignCorrectBart(
            config.model.bart,
            config.model.alignment_embedding,
            config.n_best
        )
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    
    dev_pred = get_output_format(
        config.dev_output_format,
        config.max_utt,
        config.n_best
    )

    dev_pred = run_one_epoch(
        config=config,
        model=model,
        dataloader=dev_loader,
        output=dev_pred,
        grad_update=False,
        train=False
    )
    json_saving(config.output_path + "/dev_pred.json", dev_pred)

    test_pred = get_output_format(
        config.test_output_format,
        config.max_utt,
        config.n_best
    )

    test_pred = run_one_epoch(
        config=config,
        model=model,
        dataloader=test_loader,
        output=test_pred,
        grad_update=False,
        train=False
    )
    json_saving(config.output_path + "/test_pred.json", test_pred)
    return


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    if config.seed != None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    if config.task == "training":
        train(config)
    elif config.task == "inference":
        inference(config)