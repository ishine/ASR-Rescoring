import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving, json_saving
from models.sentence_bert_lm import ErrorDetectionBert

class ErrorDetectionTraining(): 
    def __init__(self, config) -> None:
        self.config = config


    def prepare_dataset(self, file_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        parse_result = parse_json(
            file_path=file_path,
            requirements=["hyp_text", "alignment"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=False
        )
        hyp_text, all_alignment = parse_result["hyp_text"], parse_result["alignment"]

        input_ids = []
        labels = []
        num_data = len(hyp_text)
        for hyps, alignments_utt_level in tqdm(zip(hyp_text, all_alignment), total=num_data):
            for hyp, alignments in zip(hyps, alignments_utt_level):
                
                word_pieces = self.tokenizer.tokenize(hyp)

                if len(word_pieces) < self.config.max_seq_len:
                    input_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + word_pieces + ["[SEP]"]
                        )
                    )

                    operation_label_map = {"U": 0, "S": 1, "D": 1}
                    # "2" represents "[CLS]" and "[SEP]" token in labels
                    labels.append(
                        [2] + 
                        [operation_label_map[operation_token]
                        for _, __, operation_token in zip(alignments[0], alignments[1], alignments[2])
                        if operation_token in operation_label_map.keys()] +
                        [2]
                    )

        attention_masks = [
            [1]*len(row)
            for row in input_ids
        ]

        dataset = self.MyDataset(input_ids, attention_masks, labels)
        
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
        input_ids_tensor, attention_masks_tensor, labels_tensor = zip(*data)

        batch = {}

        batch["input_ids_tensor"] = pad_sequence(
            input_ids_tensor,
            batch_first=True
        )
        batch["attention_masks_tensor"] = pad_sequence(
            attention_masks_tensor,
            batch_first=True
        )
        batch["labels_tensor"] = pad_sequence(
            labels_tensor,
            batch_first=True,
            padding_value=2
        )

        return batch


    def train(self, train_dataloader, dev_dataloader):
        self.model = ErrorDetectionBert(
            bert=BertModel.from_pretrained(self.config.model)
        )

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
            input_ids_tensor = batch["input_ids_tensor"].to(self.config.device)
            attention_masks_tensor = batch["attention_masks_tensor"].to(self.config.device)
            labels_tensor = batch["labels_tensor"].to(self.config.device)

            with torch.set_grad_enabled(train_mode):
                output = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_masks_tensor
                ).squeeze(dim=2)
                
                batch_loss = self.compute_loss(output, labels_tensor)

                if train_mode:
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    
    def compute_loss(self, output, label):
        # alpha is used to balance different type of loss
        alpha = 8.3

        same_tokens_output = 1 - output[label==0]
        diff_token_output = output[label==1]

        batch_loss = torch.sum(-1 * torch.log(same_tokens_output)) \
            + alpha * torch.sum(-1 * torch.log(diff_token_output))

        return batch_loss


    class MyDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.int8)

            return input_ids_tensor, attention_masks_tensor, labels_tensor

class ErrorDetectionInference(): 
    def __init__(self, config) -> None:
        self.config = config
        self.output_json, self.hyp_text  = parse_json(
            file_path=self.config.asr_data_path,
            requirements=["all", "hyp_text"],
            max_utts=self.config.max_utts,
            flatten=False
        )

        self.UttID_and_HypID_to_SeqID = {}
        seq_id = 0
        for utt_id, utt_content in self.output_json.items():
            for hyp_id, _ in utt_content["hyp"].items():
                self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)] = seq_id
                seq_id += 1

    def prepare_dataset(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model
        )

        input_ids = []
        attention_masks = []
        LM_score_masks = []
        for hyps in self.hyp_text:
            for hyp in hyps:
                word_pieces = self.tokenizer.tokenize(hyp)
                input_ids.append(
                    self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + word_pieces + ["[SEP]"]
                    )
                )
                LM_score_masks.append(
                    [0] + [1]*len(word_pieces) + [0]
                )

        attention_masks = [
            [1]*len(row)
            for row in input_ids
        ]

        self.dataset = self.MyDataset(input_ids, attention_masks, LM_score_masks)
        
        return self.dataset


    def prepare_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
        )
        return dataloader


    def collate(self, data):
        input_ids_tensor, attention_masks_tensor, LM_score_masks_tensor, seq_idx = zip(*data)

        batch = {}

        batch["input_ids_tensor"] = pad_sequence(
            input_ids_tensor,
            batch_first=True
        )
        batch["attention_masks_tensor"] = pad_sequence(
            attention_masks_tensor,
            batch_first=True
        )
        batch["LM_score_masks_tensor"] = pad_sequence(
            LM_score_masks_tensor, 
            batch_first=True
        )
        batch["seq_id"] = list(seq_idx)

        return batch


    def scoring(self, dataloader):
        self.model = ErrorDetectionBert(
            bert=BertModel.from_pretrained(self.config.model)
        )
        checkpoint = torch.load(self.config.model_weight_path)
        self.model.load_state_dict(checkpoint)
            
        self.run_one_epoch(dataloader)


    def run_one_epoch(self, dataloader):
        self.scores = np.array([0.]*len(self.dataset))

        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, batch in loop:
            input_ids_tensor = batch["input_ids_tensor"].to(self.config.device)
            attention_masks_tensor = batch["attention_masks_tensor"].to(self.config.device)
            LM_score_masks_tensor = batch["LM_score_masks_tensor"].to(self.config.device)

            with torch.set_grad_enabled(False):

                output = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_masks_tensor
                ).squeeze(dim=2)

                batch_scores = torch.sum(
                    torch.where(LM_score_masks_tensor==1, output, torch.tensor(0, dtype=torch.float).to(self.config.device)),
                    dim=1
                )
                batch_scores = -1 * batch_scores
                np.add.at(self.scores, batch["seq_id"], batch_scores.cpu().numpy())

        for utt_id, utt_content in self.output_json.items():
            for hyp_id, hyp_content in utt_content["hyp"].items():
                seq_id = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                hyp_content["score"] = self.scores[seq_id]

        json_saving(self.config.output_path, self.output_json)


    class MyDataset(Dataset):
        def __init__(self, input_ids, attention_masks, LM_score_masks):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.LM_score_masks = LM_score_masks

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            LM_score_masks = torch.tensor(self.LM_score_masks[idx], dtype=torch.long)
            return input_ids_tensor, attention_masks_tensor, LM_score_masks, idx