import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving

class MaskedLanguageModelTraining():
    def __init__(self, config):
        self.config = config
        
        print("Parsing json data ...")
        train_origin_data = parse_json(
            file_path=config.train_data_path,
            requirements=["ref_text"],
            max_utts=self.config.max_utts,
            n_best=self.config.n_best,
            flatten=True
        )

        dev_origin_data = parse_json(
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
        self.train_dataset = self.prepare_dataset(train_origin_data)
        print("Preparing developing dataset ...")
        self.dev_dataset = self.prepare_dataset(dev_origin_data)
    
        self.train_dataloader = self.prepare_dataloader(self.train_dataset)
        self.dev_dataloader = self.prepare_dataloader(self.dev_dataset)

        print("loading model ...")
        self.model = BertForMaskedLM.from_pretrained(self.config.model)
        self.train()

    def prepare_dataset(self, origin_data):
        ref_texts = origin_data["ref_text"]

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
        batch["attention_masks"] = pad_sequence(labels_tensor, batch_first=True)

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