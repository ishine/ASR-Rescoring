import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

import numpy as np
import logging
import sys
sys.path.append("/home/chkuo/chkuo/experiment/ASR-Rescoring/src")
from models.semantic_bert import SemanticBert 

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving, json_saving
class semantic():
    def __init__(self, config):

        self.config = config
        
        if self.config.seed != None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)

        print("Parsing json data ...")
        self.train_hyp_text, self.train_hyp_cer = parse_json(
            file_path=config.train_data_path,
            requirements=["hyp_text", "hyp_cer"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        self.dev_hyp_text, self.dev_hyp_cer = parse_json(
            file_path=config.dev_data_path,
            requirements=["hyp_text", "hyp_cer"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        print("loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        print("Preparing training dataset ...")
        self.train_dataset = self.prepare_dataset(self.train_hyp_text, self.train_hyp_cer)

        print("Preparing developing dataset ...")
        self.dev_dataset = self.prepare_dataset(self.dev_hyp_text, self.dev_hyp_cer)
    
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, for_train=True)
        self.dev_dataloader = self.prepare_dataloader(self.dev_dataset)

        print("loading model ...")
        self.model = SemanticBert(bert_model=self.config.model)

        self.train()
        
    def prepare_dataset(self, hyp_texts, hyp_cers):
    
        input_ids = []
        token_type_ids = []
        labels = []

        for texts, cers in tqdm(zip(hyp_texts, hyp_cers), total=len((hyp_texts, hyp_cers))):

            for _ in (texts, cers):
                hyp_text_i = texts.pop(0)
                hyp_cer_i = cers.pop(0)

                token_seq_i = self.tokenizer.tokenize(hyp_text_i)
                for  hyp_text_j, hyp_cer_j in zip(texts, cers):
                    if hyp_cer_i != hyp_cer_j:
                        token_seq_j = self.tokenizer.tokenize(hyp_text_j)
                        if len(token_seq_i) + len(token_seq_j) < self.config.max_seq_len:
                            input_ids.append(
                                self.tokenizer.convert_tokens_to_ids(
                                    ["[CLS]"]
                                    + token_seq_i
                                    + ["[SEP]"]
                                    + token_seq_j
                                    + ["[SEP]"]
                                )
                            )

                            token_type_ids.append(
                                [0] * len(["[CLS]"] + token_seq_i + ["[SEP]"]) 
                                + [1] * len(token_seq_j + ["[SEP]"])
                            )
                            
                            labels.append([1] if hyp_cer_i < hyp_cer_j else [0])

                texts.append(hyp_text_i)
                cers.append(hyp_cer_i)

        attention_masks = [[1]*len(row) for row in input_ids]

        dataset = self.MyDataset(input_ids, token_type_ids, labels, attention_masks)
        return dataset
    
    def prepare_dataloader(self, dataset, for_train:bool = False):
        if for_train:
            shuffle=self.config.shuffle
        else:
            shuffle=False

        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker,
            shuffle=shuffle
        )
        return dataloader
    
    def collate(self, data):
        input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor = zip(*data)

        batch = {}
        batch["input_ids"] = pad_sequence(input_ids_tensor, batch_first=True)
        batch["token_type_ids"] = pad_sequence(token_type_ids_tensor, batch_first=True)
        batch["attention_masks"] = pad_sequence(attention_masks_tensor, batch_first=True)
        batch["labels"] = torch.cat(labels_tensor)
        
        return batch

    def train(self):
        train_loss_record = [0]*self.config.epoch
        dev_loss_record = [0]*self.config.epoch

        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))
            
            train_loss_record[epoch_id-1] = self.run_one_epoch(self.train_dataloader, train_mode=True)
            print("epoch ", epoch_id, " train loss: ", train_loss_record[epoch_id-1])

            dev_loss_record[epoch_id-1] = self.run_one_epoch(self.dev_dataloader, train_mode=False)
            print("epoch ", epoch_id, " dev loss: ", dev_loss_record[epoch_id-1], "\n")

            model_saving(self.config.output_path, self.model.state_dict(), epoch_id)
        
        loss_saving(
            self.config.output_path,
            {"train": train_loss_record, "dev": dev_loss_record}
        )

    def run_one_epoch(self, dataloader, train_mode: bool):
        self.model = self.model.to(self.config.device)
        
        if train_mode:
            self.model.train()
            optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.zero_grad()
        else:
            self.model.eval()

        epoch_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, batch in loop:
            input_ids = batch["input_ids"].to(self.config.device)
            token_type_ids = batch["token_type_ids"].to(self.config.device)
            attention_masks = batch["attention_masks"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            with torch.set_grad_enabled(train_mode):
                output = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                ).squeeze()

                batch_loss = self.compute_loss(output, labels)

                if train_mode:
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += batch_loss.item()
                
        return epoch_loss / len(dataloader)

    def compute_loss(self, predictions, labels):

        '''
        loss = -1*(labels*torch.log(predictions) + (1-labels)*torch.log(1-predictions))
        loss = torch.sum(loss)
        '''
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
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
            return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor


class semantic_scoring():
    def __init__(self, config):

        self.config = config

        print("Parsing json data ...")
        self.output_json, self.inference_hyp_text = parse_json(
            file_path=config.asr_data_path,
            requirements=["all", "hyp_text"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False,
            with_id=True
        )

        self.UttID_and_HypID_to_SeqID = {}
        seq_id = 0
        for utt_id, utt_content in self.output_json.items():
            for hyp_id, _ in utt_content["hyp"].items():
                self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)] = seq_id
                seq_id += 1
        
        self.corpus_len = seq_id
        
        print("loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        print("Preparing inference dataset ...")
        self.inference_dataset = self.prepare_dataset(self.inference_hyp_text)

        self.inference_dataloader = self.prepare_dataloader(self.inference_dataset, for_train=False)

        print("loading model ...")
        self.model = SemanticBert(bert_model=self.config.model)
        checkpoint = torch.load(self.config.model_weight_path)
        self.model.load_state_dict(checkpoint)

        self.scoring(self.inference_dataloader)
        
    def prepare_dataset(self, hyp_texts):
    
        input_ids = []
        token_type_ids = []
        seq_id = []

        for texts in tqdm(hyp_texts, total=len(hyp_texts)):

            for _ in (texts):
                hyp_text_i = texts.pop(0)
                token_seq_i = self.tokenizer.tokenize(hyp_text_i)
                for  hyp_text_j in texts:
                    token_seq_j = self.tokenizer.tokenize(hyp_text_j)

                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"]
                            + token_seq_i
                            + ["[SEP]"]
                            + token_seq_j
                            + ["[SEP]"]
                        )
                    )

                    token_type_ids.append(
                        [0] * len(["[CLS]"] + token_seq_i + ["[SEP]"]) 
                        + [1] * len(token_seq_j + ["[SEP]"])
                    )
                    seq_id_i = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                    seq_id_j = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                    seq_id.append((seq_id_i, seq_id_j)) 

                texts.append(hyp_text_i)

        attention_masks = [[1]*len(row) for row in input_ids]

        dataset = self.MyDataset(input_ids, token_type_ids, attention_masks, seq_id)
        return dataset
    
    def prepare_dataloader(self, dataset, for_train:bool = False):
        if for_train:
            shuffle=self.config.shuffle
        else:
            shuffle=False

        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker,
            shuffle=shuffle
        )
        return dataloader
    
    def collate(self, data):
        input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, seq_id_i, seq_id_j = zip(*data)

        batch = {}
        batch["input_ids"] = pad_sequence(input_ids_tensor, batch_first=True)
        batch["token_type_ids"] = pad_sequence(token_type_ids_tensor, batch_first=True)
        batch["attention_masks"] = pad_sequence(attention_masks_tensor, batch_first=True)
        batch["seq_id_i"] = seq_id_i
        batch["seq_id_j"] = seq_id_j

        return batch

    def scoring(self, inference_dataloader):
        self.scores = np.array([0.]*self.corpus_len)

        self.model = self.model.to(self.config.device)
        self.model.eval()

        loop = tqdm(enumerate(inference_dataloader), total=len(inference_dataloader))
        for _, batch in loop:
            input_ids = batch["input_ids"].to(self.config.device)
            token_type_ids = batch["token_type_ids"].to(self.config.device)
            attention_masks = batch["attention_masks"].to(self.config.device)
            seq_id_i = batch["seq_id_i"]
            seq_id_j = batch["seq_id_j"]

            with torch.set_grad_enabled(False):
                semantic_score = self.model(
                    input_ids = input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_masks
                ).squeeze(dim=1)

                # seq_id 代表 batch中每個seq屬於「哪個utt中的哪個hypothesis sentence的獨一無二的id」
                np.add.at(self.scores, list(seq_id_i), semantic_score.cpu().numpy())
                np.add.at(self.scores, list(seq_id_j), (1-semantic_score).cpu().numpy())

        for utt_id, utt_content in self.output_json.items():
            for hyp_id, hyp_content in utt_content["hyp"].items():
                seq_id = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                hyp_content["score"] = self.scores[seq_id]

        json_saving(self.config.output_path, self.output_json)


    class MyDataset(Dataset):
        def __init__(self, input_ids, token_type_ids, attention_masks, seq_id):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_masks = attention_masks
            self.seq_id = seq_id

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            token_type_ids_tensor = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            (seq_id_i, seq_id_j) = self.seq_id[idx]
            return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, seq_id_i, seq_id_j
