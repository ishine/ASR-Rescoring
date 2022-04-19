import abc
import json
import numpy as np
import logging
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from models.Fohr21INTERSPEECH import BERTsem, BERTalsem 
from util.dataparser import DataParser
from util.saving import json_saving, loss_saving, model_saving
from util.freezer import freeze_by_name, unfreeze_by_name


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, config) -> None:
        
        print("Setting trainer ...")
        self.config = config
        logging.basicConfig(filename=self.config.output_path + "/training.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        self.seed = config.seed
        self.device = config.device
        
        self.train_data_file = config.train_data_file
        self.dev_data_file = config.dev_data_file
        self.output_path = config.output_path

        self.preprocess_config = config.preprocess
        self.model_config = config.model
        self.dataloader_config = config.dataloader
        self.opt_config = config.opt

        if self.seed != None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
    
    def parse_data(self, data_path):
        data_parser = DataParser(
            json_file_path=data_path,
            max_utts=self.preprocess_config.max_utts,
            n_best=self.preprocess_config.n_best
        )
        data = data_parser.parse()
        
        return data
    
    @abc.abstractmethod
    def prepare_dataset(self):
        return NotImplementedError

    def prepare_dataloader(self, dataset, for_train:bool):
        if for_train:
            shuffle=self.dataloader_config.shuffle
        else:
            shuffle=False

        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.dataloader_config.batch_size,
            num_workers=self.dataloader_config.num_worker,
            shuffle=shuffle
        )
        return dataloader

    @abc.abstractmethod
    def load_model(self):
        return NotImplementedError

    @abc.abstractmethod
    def run_one_epoch(self):
        return NotImplementedError

    def train(self):

        print("Parsing train data ...")
        self.train_data = self.parse_data(self.train_data_file)

        print("Parsing dev data ...")
        self.dev_data = self.parse_data(self.dev_data_file)
        
        print("Loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_config.bert
        )

        print("Preparing train dataset and dataloader ...")
        self.train_dataset = self.prepare_dataset(self.train_data.utt_set)
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, for_train=True)

        print("Preparing dev dataset and dataloader ...")
        self.dev_dataset = self.prepare_dataset(self.dev_data.utt_set)
        self.dev_dataloader = self.prepare_dataloader(self.dev_dataset, for_train=False)

        print("Loading model ...")
        self.model = self.load_model()

        print("Start training ...")
        step = 1000
        loss_record = {
            "train": {"every_epoch": [], f"every_{step}_batch": []},
            "dev": {"every_epoch": [], f"every_{step}_batch": []}
        }

        for epoch_id in range(1, self.opt_config.epoch+1):
            if self.config.resume.epoch!= None and epoch_id <= self.config.resume.epoch:
                loss_record = json.load(
                    open(self.output_path + "/loss.json", "r", encoding="utf-8")
                )

                print("skip epoch ", epoch_id)
                continue

            print("Epoch {}/{}".format(epoch_id, self.opt_config.epoch))
            
            if epoch_id <= self.opt_config.bert_freeze_epoch:
                freeze_by_name(self.model, "bert")
            else:
                unfreeze_by_name(self.model, "bert")

            epoch_loss, batch_loss = self.run_one_epoch(self.train_dataloader, train_mode=True)
            loss_record["train"]["every_epoch"].append(epoch_loss)
            loss_record["train"][f"every_{step}_batch"] += batch_loss
            print("Epoch ", epoch_id, " train loss: ", epoch_loss, "\n")

            epoch_loss, batch_loss = self.run_one_epoch(self.dev_dataloader, train_mode=False)
            loss_record["dev"]["every_epoch"].append(epoch_loss)
            loss_record["dev"][f"every_{step}_batch"] += batch_loss
            print("Epoch ", epoch_id, " dev loss: ", epoch_loss, "\n")

            model_saving(self.output_path, self.model.state_dict(), epoch_id)
        
            json_saving(
                self.output_path + "/loss.json" ,
                loss_record
            )

        print("Training finish.")


class BERTsemTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_dataset(self, data):
        input_ids = []
        token_type_ids = []
        labels = []
        for utt in tqdm(data, total=len(data)): 

            for _ in range(len(utt.hyps)):

                hyp_i = utt.hyps.pop(0)
                for hyp_j in utt.hyps:
                    if hyp_i.cer == hyp_j.cer:
                        continue

                    hyp_i_token_seq = self.tokenizer.tokenize(hyp_i.text)
                    hyp_j_token_seq = self.tokenizer.tokenize(hyp_j.text)
                    if len(hyp_i_token_seq + hyp_j_token_seq) > self.preprocess_config.max_seq_len:
                        continue

                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + hyp_i_token_seq + ["[SEP]"]
                            + hyp_j_token_seq + ["[SEP]"]
                        )
                    )
                    token_type_ids.append(
                        [0] * len(["[CLS]"] + hyp_i_token_seq + ["[SEP]"])
                        + [1] * len(hyp_j_token_seq + ["[SEP]"])
                    )
                    labels.append(1 if hyp_i.cer < hyp_j.cer else 0)

                utt.hyps.append(hyp_i)

        attention_masks = [[1]*len(row) for row in input_ids]
        
        return self.MyDataset(input_ids, token_type_ids, labels, attention_masks)
    
    def collate(self, data):
        input_ids_tensors, token_type_ids_tensors, labels_tensors, attention_masks_tensors = zip(*data)

        input_ids_tensor = pad_sequence(input_ids_tensors, batch_first=True)
        token_type_ids_tensor = pad_sequence(token_type_ids_tensors, batch_first=True)
        attention_masks_tensor = pad_sequence(attention_masks_tensors, batch_first=True)
        labels_tensor = torch.stack(labels_tensors)

        return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor
    
    def load_model(self):
        self.model = BERTsem(self.model_config)
        if self.config.resume.model_weight_path != None:
            checkpoint = torch.load(self.config.resume.model_weight_path)
            self.model.load_state_dict(checkpoint)
        return self.model
    
    def run_one_epoch(self, dataloader, train_mode: bool):
        self.model = self.model.to(self.device)
        
        if train_mode:
            self.model.train()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.opt_config.lr)
            optimizer.zero_grad()
        else:
            self.model.eval()

        epoch_loss = 0
        batch_loss = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, (input_ids, token_type_ids, attention_masks, labels) in loop:
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(train_mode):
                output = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                )
                loss = self.compute_loss(output, labels)
                batch_loss.append(loss)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item()
                
        return epoch_loss / len(dataloader), batch_loss
    
    def compute_loss(self, predictions, labels):
        BCE_loss_fn = torch.nn.BCELoss(reduction='mean')
        loss = BCE_loss_fn(predictions, labels)
        return loss

    class MyDataset(Dataset):
        def __init__(self, input_ids, token_type_ids, labels, attention_masks):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.labels = labels
            self.attention_masks = attention_masks
            
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            token_type_ids_tensor = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return input_ids_tensor, token_type_ids_tensor, labels_tensor, attention_masks_tensor


class BERTalsemTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_dataset(self, data):
        input_ids = []
        token_type_ids = []
        scores = []
        labels = []
        
        for utt in tqdm(data, total=len(data)): 

            for _ in range(len(utt.hyps)):

                hyp_i = utt.hyps.pop(0)
                for hyp_j in utt.hyps:
                    if hyp_i.cer == hyp_j.cer:
                        continue

                    hyp_i_token_seq = self.tokenizer.tokenize(hyp_i.text)
                    hyp_j_token_seq = self.tokenizer.tokenize(hyp_j.text)
                    if len(hyp_i_token_seq + hyp_j_token_seq) > self.preprocess_config.max_seq_len:
                        continue

                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + hyp_i_token_seq + ["[SEP]"]
                            + hyp_j_token_seq + ["[SEP]"]
                        )
                    )
                    token_type_ids.append(
                        [0] * len(["[CLS]"] + hyp_i_token_seq + ["[SEP]"])
                        + [1] * len(hyp_j_token_seq + ["[SEP]"])
                    )
                    scores.append([hyp_i.score / len(hyp_i.text), hyp_j.score / len(hyp_j.text)])
                    labels.append(1 if hyp_i.cer < hyp_j.cer else 0)

                utt.hyps.append(hyp_i)

        attention_masks = [[1]*len(row) for row in input_ids]
        
        return self.MyDataset(input_ids, token_type_ids, scores, labels, attention_masks)
    
    def collate(self, data):
        input_ids_tensors, token_type_ids_tensors, scores_tensors, \
            labels_tensors, attention_masks_tensors = zip(*data)

        input_ids_tensor = pad_sequence(input_ids_tensors, batch_first=True)
        token_type_ids_tensor = pad_sequence(token_type_ids_tensors, batch_first=True)
        attention_masks_tensor = pad_sequence(attention_masks_tensors, batch_first=True)
        scores_tensor = torch.stack(scores_tensors)
        labels_tensor = torch.stack(labels_tensors)

        return input_ids_tensor, token_type_ids_tensor, \
            attention_masks_tensor, scores_tensor, labels_tensor
    
    def load_model(self):
        self.model = BERTalsem(self.model_config)
        if self.config.resume.model_weight_path != None:
            checkpoint = torch.load(self.config.resume.model_weight_path)
            self.model.load_state_dict(checkpoint)
        return self.model
    
    def run_one_epoch(self, dataloader, train_mode: bool):
        self.model = self.model.to(self.device)
        
        if train_mode:
            self.model = self.model.train()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.opt_config.lr)
            optimizer.zero_grad()
        else:
            self.model = self.model.eval()

        epoch_loss = 0
        batch_loss_record = []
        batch_loss = 0
        loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader))
        for batch_id, (input_ids, token_type_ids, attention_masks, scores, labels) in loop:
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            scores = scores.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(train_mode):
                output = self.model(
                    self.device,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    scores = scores.to(self.device),
                ).squeeze(dim=1)
                loss = self.compute_loss(output, labels)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            batch_loss += loss.item()
            if batch_id % self.config.batch_loss_rescord_step == 0:
                batch_loss /= self.config.batch_loss_rescord_step
                logging.info(f"batch {batch_id} loss: " + str(batch_loss))
                batch_loss_record.append(batch_loss)
                batch_loss = 0

            epoch_loss += loss.item()
                
        return epoch_loss / len(dataloader), batch_loss_record
    
    def compute_loss(self, predictions, labels):
        BCE_loss_fn = torch.nn.BCELoss(reduction='mean')
        loss = BCE_loss_fn(predictions, labels)
        return loss
        
    class MyDataset(Dataset):
        def __init__(self, input_ids, token_type_ids, scores, labels, attention_masks):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.scores = scores
            self.labels = labels
            self.attention_masks = attention_masks
            
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            token_type_ids_tensor = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            scores_tensor = torch.tensor(self.scores[idx], dtype=torch.float)
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return input_ids_tensor, token_type_ids_tensor, scores_tensor, \
                labels_tensor, attention_masks_tensor
