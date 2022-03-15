import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from models.sentence_bert_lm import SentenceBertLM
from util.parse_json import parse_json
from util.saving import model_saving, loss_saving, json_saving

class MD_MWER_Training():
    def __init__(self, config):
        self.config = config
        
        print("Parsing json data ...")
        self.train_hyp_text, self.train_hyp_am_score, self.train_hyp_cer = parse_json(
            file_path=config.train_am_path,
            requirements=["hyp_text", "hyp_score", "hyp_cer"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        self.train_hyp_lm_score = parse_json(
            file_path=config.train_lm_path,
            requirements=["hyp_score"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        self.dev_hyp_text, self.dev_hyp_am_score, self.dev_hyp_cer = parse_json(
            file_path=config.dev_am_path,
            requirements=["hyp_text", "hyp_score", "hyp_cer"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        self.dev_hyp_lm_score = parse_json(
            file_path=config.dev_lm_path,
            requirements=["hyp_score"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )

        print("loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        print("Preparing training dataset ...")
        self.train_dataset = self.prepare_dataset(
            self.train_hyp_text,
            self.train_hyp_am_score,
            self.train_hyp_lm_score,
            self.train_hyp_cer
        )
        print("Preparing developing dataset ...")
        self.dev_dataset = self.prepare_dataset(
            self.dev_hyp_text,
            self.dev_hyp_am_score,
            self.dev_hyp_lm_score,
            self.dev_hyp_cer
        )
    
        self.train_dataloader = self.prepare_dataloader(self.train_dataset)
        self.dev_dataloader = self.prepare_dataloader(self.dev_dataset)

        print("loading model ...")
        self.model = SentenceBertLM(
            bert=BertModel.from_pretrained(self.config.model)
        )

        self.train(self.train_dataloader, self.dev_dataloader)

    def prepare_dataset(self, hyp_texts, hyp_am_scores, hyp_lm_scores, hyp_cers):
       
        input_ids = []
        attention_masks = []
        for hyps in tqdm(hyp_texts, total=len(hyp_texts)):
            
            batch_ids = []
            batch_attention_masks = []
            
            for hyp in hyps:
                word_pieces = self.tokenizer.tokenize(hyp)

                if len(word_pieces) > self.config.max_seq_len:
                    break
                
                batch_ids.append(
                    self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + word_pieces + ["[SEP]"]
                    )
                )

                batch_attention_masks.append(
                    [1] * len(["[CLS]"] + word_pieces + ["[SEP]"])
                )

            if len(batch_ids) == self.config.n_best:
                input_ids.append(batch_ids)
                attention_masks.append(batch_attention_masks)

        dataset = self.MyDataset(input_ids, attention_masks, hyp_am_scores, hyp_lm_scores, hyp_cers)
        
        return dataset

    def prepare_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
            num_workers=3
        )
        return dataloader

    def collate(self, data):
        input_ids_tensor_sets, attention_masks_tensor_sets, asr_scores_tensor_sets,\
            lm_scores_tensor_sets, hyp_cer_tensor_sets = zip(*data)

        batch = {}

        batch["input_ids"] = []
        for input_ids_tensor_set in input_ids_tensor_sets:
            batch["input_ids"] += input_ids_tensor_set
        batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True)

        batch["attention_masks"] = []
        for attention_masks_tensor_set in attention_masks_tensor_sets:
            batch["attention_masks"] += attention_masks_tensor_set
        batch["attention_masks"] = pad_sequence(batch["attention_masks"], batch_first=True)
        
        batch["asr_scores"] = torch.stack(asr_scores_tensor_sets)
        batch["lm_scores"] = torch.stack(lm_scores_tensor_sets)
        batch["hyp_cer"] = torch.stack(hyp_cer_tensor_sets)

        return batch

    def train(self, train_dataloader, dev_dataloader):

        train_loss_record = [0]*self.config.epoch
        dev_loss_record = [0]*self.config.epoch

        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))
            
            train_loss_record[epoch_id-1] = self.run_one_epoch(train_dataloader, train_mode=True)
            print("epoch ", epoch_id, " train loss: ", train_loss_record[epoch_id-1])

            dev_loss_record[epoch_id-1] = self.run_one_epoch(dev_dataloader, train_mode=False)
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
            attention_masks = batch["attention_masks"].to(self.config.device)
            asr_scores = batch["asr_scores"].to(self.config.device)
            lm_scores = batch["lm_scores"].to(self.config.device)
            hyp_cer = batch["hyp_cer"].to(self.config.device)

            with torch.set_grad_enabled(train_mode):
                predicted_LM_scores = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                ).squeeze(dim=1)

                predicted_LM_scores = torch.reshape(
                    predicted_LM_scores,
                    (int(len(predicted_LM_scores)/self.config.n_best), self.config.n_best)
                )

                batch_MWER_loss = self.compute_MWER_loss(asr_scores, predicted_LM_scores, hyp_cer)

                bath_MD_loss = self.compute_MD_loss(predicted_LM_scores, lm_scores)

                batch_loss = batch_MWER_loss + self.config.MD_loss_weight * bath_MD_loss

                if train_mode:
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def compute_MD_loss(self, predicted_LM_scores, label_LM_scores):
        mse_loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = mse_loss_fn(predicted_LM_scores, label_LM_scores)
        return loss
    
    def compute_MWER_loss(self, asr_scores, LM_scores, hyp_cer):
        final_scores = asr_scores + LM_scores

        probility = torch.softmax(-1*final_scores, dim=1)

        average_cer = torch.sum(hyp_cer,dim=1) / self.config.n_best
        average_cer = average_cer.unsqueeze(dim=1)

        batch_loss = torch.sum(torch.mul(probility, (hyp_cer - average_cer)))

        return batch_loss

    class MyDataset(Dataset):
        def __init__(self, input_ids, attention_masks, asr_scores, lm_scores, hyp_cers):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.asr_scores = asr_scores
            self.lm_scores = lm_scores
            self.hyp_cers = hyp_cers

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = [torch.tensor(ids, dtype=torch.long) 
                for ids in self.input_ids[idx]]
            
            attention_masks_tensor = [torch.tensor(mask, dtype=torch.long) 
                for mask in self.attention_masks[idx]]

            asr_scores_tensor = torch.tensor(
                [asr_score for asr_score in self.asr_scores[idx]],
                dtype=torch.float
            )
            
            lm_scores_tensor = torch.tensor(
                [lm_score for lm_score in self.lm_scores[idx]],
                dtype=torch.float
            )

            hyp_cer_tensor = torch.tensor(
                [hyp_cer for hyp_cer in self.hyp_cers[idx]],
                dtype=torch.float
            ) 

            return input_ids_tensor, attention_masks_tensor, \
                asr_scores_tensor, lm_scores_tensor, hyp_cer_tensor
