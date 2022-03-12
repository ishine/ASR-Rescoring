import os
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

from util.parse_json import parse_json
from util.saving import model_saving, loss_saving
from models.sentence_bert_lm import SentenceBertLM

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


class MLMDistill():
    def __init__(self, config) -> None:
        self.config = config
        self.model = SentenceBertLM(
            bert=BertModel.from_pretrained(config.model)
        )
        self.tokenizer = load_tokenizer(config.model)
         
    def prepare_train_set(self):
        corpus = get_recog_data(self.config.train_data_path, self.config.text_type, self.config.max_utts)
        labels = get_recog_data(self.config.train_data_path, "hyp_score", self.config.max_utts)

        # 過濾過長的seq，並把每個seq複製和它長度一樣的次數
        ids_list = []
        label_list = []
        attention_mask_list = []

        for utt_hyps_text, utt_hyps_score in zip(corpus, labels):
            for hyp_text, hyp_score in zip(utt_hyps_text, utt_hyps_score):

                token_seq = self.tokenizer.tokenize(hyp_text)
                if len(token_seq) < self.config.max_seq_len:
                    ids_list.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )

                    label_list.append(hyp_score)

                    attention_mask_list.append([1]*len(["[CLS]"] + token_seq + ["[SEP]"]))
        
        self.train_set = self.trainDataset(ids_list, label_list, attention_mask_list)
        
        return self.train_set

    def prepare_train_loader(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
        )
        return self.train_loader

    def collate(self, data):
        id_tensor, label, mask_tensor = zip(*data)
        
        id_tensor = pad_sequence(id_tensor, batch_first=True)
        mask_tensor = pad_sequence(mask_tensor, batch_first=True)
        label_tensor = torch.tensor(label)
        return id_tensor, label_tensor, mask_tensor

    def train(self):
        train_loss_record = [0]*self.config.epoch
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))

            self.model = self.model.to(self.config.device)
            self.model.train()
            
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for _, (id_tensor, label_tensor, mask_tensor) in loop:

                id_tensor = id_tensor.to(self.config.device)
                label_tensor = label_tensor.to(self.config.device)
                mask_tensor = mask_tensor.to(self.config.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    output = self.model(
                        input_ids=id_tensor,
                        attention_mask=mask_tensor
                    )
                    output = torch.squeeze(output)
                    loss = MSELoss()(output, label_tensor)
                    loss.backward()
                    optimizer.step()
                
                train_loss_record[epoch_id-1] += loss.item()
            
            train_loss_record[epoch_id-1] /= len(self.train_loader)
            print("epoch ", epoch_id, "loss: ", train_loss_record[epoch_id-1])

            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.output_path, 'checkpoint_{}.pth'.format(epoch_id))
            )

            with open(self.config.output_path + "loss.txt", "w") as f:
                f.write("epoch loss: \n")
                f.write(" ".join([str(loss) for loss in train_loss_record]))

    class trainDataset(Dataset):
        def __init__(self, ids_list, label_list, attention_mask_list):
            self.ids_list = ids_list
            self.label_list = label_list
            self.attention_mask_list = attention_mask_list

        def __len__(self):
            return len(self.ids_list)
        
        def __getitem__(self, idx):
            ids_tensor = torch.tensor(self.ids_list[idx], dtype=torch.long)
            label = self.label_list[idx]
            attention_mask_tensor = torch.tensor(self.attention_mask_list[idx], dtype=torch.long)
            return ids_tensor, label, attention_mask_tensor
