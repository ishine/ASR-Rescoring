import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving, json_saving

class MaskedLanguageModelTraining():
    def __init__(self, config):
        self.config = config
        
        print("Parsing json data ...")
        self.train_ref_text = parse_json(
            file_path=config.train_data_path,
            requirements=["ref_text"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=True
        )

        self.dev_ref_text = parse_json(
            file_path=config.dev_data_path,
            requirements=["ref_text"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=True
        )

        print("loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        print("Preparing training dataset ...")
        self.train_dataset = self.prepare_dataset(self.train_ref_text)
        print("Preparing developing dataset ...")
        self.dev_dataset = self.prepare_dataset(self.dev_ref_text)
    
        self.train_dataloader = self.prepare_dataloader(self.train_dataset)
        self.dev_dataloader = self.prepare_dataloader(self.dev_dataset)

        print("loading model ...")
        self.model = BertForMaskedLM.from_pretrained(self.config.model)

        self.train()

    def prepare_dataset(self, ref_texts):
    
        labels = []
        input_ids = []

        for ref in tqdm(ref_texts, total=len(ref_texts)):

            token_seq = self.tokenizer.tokenize(ref)
            if len(token_seq) < self.config.max_seq_len:

                for mask_pos in range(len(token_seq)):
                    labels.append(
                        self.tokenizer.convert_tokens_to_ids(
                                ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )

                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"]
                            + token_seq[:mask_pos] + ["[MASK]"] + token_seq[mask_pos+1:]
                            + ["[SEP]"]
                        )
                    )

        attention_masks = [[1]*len(row) for row in input_ids]

        dataset = self.MyDataset(input_ids, labels, attention_masks)
        return dataset
    
    def prepare_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker
        )
        return dataloader
    
    def collate(self, data):
        input_ids_tensor, labels_tensor, attention_masks_tensor = zip(*data)

        batch = {}
        batch["input_ids"] = pad_sequence(input_ids_tensor, batch_first=True)
        batch["labels"] = pad_sequence(labels_tensor, batch_first=True)
        batch["attention_masks"] = pad_sequence(attention_masks_tensor, batch_first=True)

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
            attention_masks = batch["attention_masks"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            with torch.set_grad_enabled(train_mode):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    labels=labels
                )

                if train_mode:
                    output.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += output.loss.item()
                
        return epoch_loss / len(dataloader)

    class MyDataset(Dataset):
        def __init__(self, input_ids, labels, attention_masks):
            self.input_ids = input_ids
            self.labels = labels
            self.attention_masks = attention_masks

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return input_ids_tensor, labels_tensor, attention_masks_tensor

class PLLScoring():
    def __init__(self, config):
        self.config = config

        print("Parsing json data ...")
        self.output_json = parse_json(
            file_path=config.asr_data_path,
            requirements=["all"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
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
        self.inference_dataset = self.prepare_dataset()

        self.inference_dataloader = self.prepare_dataloader()

        print("loading model ...")
        self.model = BertForMaskedLM.from_pretrained(self.config.model)
        checkpoint = torch.load(self.config.model_weight_path)
        self.model.load_state_dict(checkpoint)
        
        self.scoring()

    def prepare_dataset(self):
        input_ids = []
        attention_masks = []
        masked_token_pos = []
        masked_token_ids = []
        seq_id = []

        # 把每個seq複製和它長度一樣的次數
        for utt_id, utt_content in self.output_json.items():
            for hyp_id, hyp_content in utt_content["hyp"].items():

                token_seq = self.tokenizer.tokenize(hyp_content["text"])
                
                for mask_pos in range(len(token_seq)):
                    seq_id.append(self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)])

                    masked_token_pos.append(mask_pos+1)

                    masked_token_ids.append(self.tokenizer.convert_tokens_to_ids(token_seq[mask_pos]))
                    
                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"]
                            + token_seq[:mask_pos] + ["[MASK]"] + token_seq[mask_pos+1:]
                            + ["[SEP]"]
                        )
                    )

        attention_masks = [[1]*len(row) for row in input_ids]
        
        dataset = self.MyDataset(
            input_ids, attention_masks, masked_token_pos, masked_token_ids, seq_id)
        
        return dataset
    
    def prepare_dataloader(self):
        dataloader = DataLoader(
            dataset=self.inference_dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker
        )
        return dataloader

    def collate(self, data):
        input_ids_tensor, attention_masks_tensor, masked_token_pos,\
                masked_token_ids, seq_id = zip(*data)

        batch = {}
        batch["input_ids"] = pad_sequence(input_ids_tensor, batch_first=True)
        batch["attention_masks"] = pad_sequence(attention_masks_tensor, batch_first=True)
        batch["masked_token_pos"] = masked_token_pos
        batch["masked_token_ids"] = masked_token_ids
        batch["seq_id"] = seq_id

        return batch

    def scoring(self):
        self.scores = np.array([0.]*self.corpus_len)

        self.model = self.model.to(self.config.device)
        self.model.eval()

        loop = tqdm(enumerate(self.inference_dataloader), total=len(self.inference_dataloader))
        for _, batch in loop:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_masks = batch["attention_masks"].to(self.config.device)
            masked_token_pos = batch["masked_token_pos"]
            masked_token_ids = batch["masked_token_ids"]
            seq_id = batch["seq_id"]

            # 每個seq在這個batch中的位置，index從0開始，所以第一個seq的位置=0
            seq_pos_in_batch = list(range(len(input_ids)))

            with torch.set_grad_enabled(False):
                # output_logits -> size: [batch size(seq num), seq length(token num), vocab size]
                output_logits = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_masks
                ).logits

                # 取出這個batch中每個seq個被mask的token的位置對應的output logits
                output_logits = output_logits[seq_pos_in_batch, masked_token_pos,:]

                # 對最後一維，也就是長度是vocab size的那維做softmax，再把全部數值取log
                output_score = output_logits.log_softmax(dim=-1)

                # 利用masked_token_id找出 masked token在vocab size那維的位置
                output_score = output_score[seq_pos_in_batch, masked_token_ids]

                # seq_id 代表 batch中每個seq屬於「哪個utt中的哪個hypothesis sentence的獨一無二的id」
                np.add.at(self.scores, list(seq_id), output_score.cpu().numpy())
        
        for utt_id, utt_content in self.output_json.items():
            for hyp_id, hyp_content in utt_content["hyp"].items():
                seq_id = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                hyp_content["score"] = self.scores[seq_id]

        json_saving(self.config.output_path, self.output_json)

    class MyDataset(Dataset):
        def __init__(self, input_ids, attention_masks, masked_token_pos, masked_token_ids, seq_id):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.masked_token_pos = masked_token_pos
            self.masked_token_ids = masked_token_ids
            self.seq_id = seq_id
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        
            return input_ids_tensor, attention_masks_tensor, self.masked_token_pos[idx],\
                self.masked_token_ids[idx], self.seq_id[idx]
