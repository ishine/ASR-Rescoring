import os
import json
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertModel
from tqdm import tqdm
from inference import get_recog_data
from models.sentence_bert_lm import SentenceBertLM

def get_corpus(data_path, type, max_utts):
    utt_count = 0
    corpus = []
    json_data = json.load(open(data_path, "r", encoding="utf-8"))
    for recog_content in json_data.values():
        if type == "ref":
            corpus.append(recog_content["ref"])
            utt_count += 1
            if utt_count == max_utts:
                break
    return corpus


def load_model(model_name, model_weight_path=None):
    model = BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    if model_weight_path != None:
        checkpoint = torch.load(model_weight_path)
        model.load_state_dict(checkpoint)
    return model


def load_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    return tokenizer


def mask_sentence(tokenizer, sentence_wordpiece, mask_pos):
    mask_token = tokenizer.convert_tokens_to_ids("[MASK]")
    output = copy.deepcopy(sentence_wordpiece)
    output[mask_pos] = mask_token
    return output


def collate(data):
    id_tensor, label_tensor, mask_tensor = zip(*data)
    
    id_tensor = pad_sequence(id_tensor, batch_first=True)
    label_tensor = pad_sequence(label_tensor, batch_first=True)
    mask_tensor = pad_sequence(mask_tensor, batch_first=True)    
    return id_tensor, label_tensor, mask_tensor


class DomainAdaptation():
    def __init__(self, config):
        self.config = config
        if config.model_weight_path != None:
            self.model = load_model(config.model, config.model_weight_path)
        else:
            self.model = load_model(config.model)
        self.tokenizer = load_tokenizer(config.model)
        self.corpus = get_corpus(config.train_data_path, config.text_type, config.max_utts)
        
    def prepare_train_set(self):
        # 過濾過長的seq，並把每個seq複製和它長度一樣的次數
        label_set = []
        id_set = []
        mask_set = []
        for seq in self.corpus:
            token_seq = self.tokenizer.tokenize(seq)
            if len(token_seq) < self.config.max_seq_len:
                for mask_pos in range(len(token_seq)):
                    label_set.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )
                    masked_token_seq = mask_sentence(self.tokenizer, token_seq, mask_pos)
                    id_set.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + masked_token_seq + ["[SEP]"]
                        )
                    )
                    mask_set.append([1]*len(["[CLS]"] + token_seq + ["[SEP]"]))
        self.train_set = self.trainDataset(id_set, label_set, mask_set)
        
        return self.train_set

    def prepare_train_loader(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            collate_fn=collate,
            batch_size=self.config.batch_size,
        )
        return self.train_loader

    def train(self):
        train_loss_record = [0]*self.config.epoch
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))

            self.model = self.model.to(self.config.device)
            self.model.train()
            
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for _, (id_tensor, label_tensor, mask_tensor) in loop:
                
                id_tensor = id_tensor.to(self.config.device)
                label_tensor = label_tensor.to(self.config.device)
                mask_tensor = mask_tensor.to(self.config.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(
                        input_ids=id_tensor,
                        labels=label_tensor,
                        attention_mask=mask_tensor
                    )
                    loss = outputs[0]
                    loss.backward()
                    self.optimizer.step()
                
                train_loss_record[epoch_id-1] += loss.item()
            
            train_loss_record[epoch_id-1] /= len(self.train_loader)
            print("epoch ", epoch_id, "loss: ", train_loss_record[epoch_id-1])

            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.output_path, 'checkpoint_{}.pth'.format(epoch_id+8))
            )

            with open(self.config.output_path + "loss.txt", "w") as f:
                f.write("epoch loss: \n")
                f.write(" ".join([str(loss) for loss in train_loss_record]))

    class trainDataset(Dataset):
        def __init__(self, id_set, label_set, mask_set):
            self.id_set = id_set
            self.label_set = label_set
            self.mask_set = mask_set

        def __len__(self):
            return len(self.id_set)
        
        def __getitem__(self, idx):
            id_tensor = torch.tensor(self.id_set[idx], dtype=torch.long)
            label_tensor = torch.tensor(self.label_set[idx], dtype=torch.long)
            mask_tensor = torch.tensor(self.mask_set[idx], dtype=torch.long)
            return id_tensor, label_tensor, mask_tensor


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


class MWERTraining(): 
    def __init__(self, config) -> None:
        self.config = config
        self.model = SentenceBertLM(
            bert=BertModel.from_pretrained(config.model)
        )
        self.tokenizer = load_tokenizer(config.model)
         
    def prepare_dataset(self, data_path):
        corpus = get_recog_data(data_path, "hyp_text", self.config.max_utts, flatten=False)
        ASR_score_list = get_recog_data(data_path, "hyp_score", self.config.max_utts)
        cer_list = get_recog_data(data_path, "cer", self.config.max_utts)
        
        ids_list = []
        attention_mask_list = []
        utt_id_list = []

        for utt_id, utt_hyps_text in enumerate(corpus):
            for hyp_text in utt_hyps_text:
                token_seq = self.tokenizer.tokenize(hyp_text)
                if len(token_seq) < self.config.max_seq_len:
                    ids_list.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + token_seq + ["[SEP]"]
                        )
                    )

                    attention_mask_list.append([1]*len(["[CLS]"] + token_seq + ["[SEP]"]))
                    
                    utt_id_list.append(utt_id)

        dataset = self.trainDataset(ids_list, attention_mask_list, ASR_score_list, cer_list, utt_id_list)
        
        return dataset

    def prepare_train_loader(self, dataset):
        self.train_loader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
        )
        return self.train_loader

    def collate(self, data):
        id_list, mask_list, ASR_score_list, cer_list, utt_ids = zip(*data)
        tmp = []
        for id in id_list:
            tmp += id
        id_list = tmp

        tmp = []
        for mask in mask_list:
            tmp += mask
        mask_list = tmp

        tmp = []
        for ASR_score in ASR_score_list:
            tmp += ASR_score
        ASR_score_list = tmp

        tmp = []
        for cer in cer_list:
            tmp += cer
        cer_list = tmp

        tmp = []
        for utt in utt_ids:
            tmp += utt
        utt_ids = tmp

        id_tensor = pad_sequence([torch.tensor(id, dtype=torch.long) for id in id_list ], batch_first=True)
        mask_tensor = pad_sequence([torch.tensor(mask, dtype=torch.long) for mask in mask_list ], batch_first=True)
        ASR_score_tensor = torch.tensor(ASR_score_list)
        cer_tensor = torch.tensor(cer_list)

        return id_tensor, mask_tensor, ASR_score_tensor, cer_tensor, utt_ids

    def train(self, train_dataloader, dev_loader):
        train_loss_record = [0]*self.config.epoch
        dev_loss_record = [0]*self.config.epoch

        for epoch_id in range(1, self.config.epoch+1):
            print("Epoch {}/{}".format(epoch_id, self.config.epoch))
            
            train_loss_record[epoch_id-1] = self.train_one_epoch(train_dataloader)
            print("epoch ", epoch_id, "train loss: ", train_loss_record[epoch_id-1])

            dev_loss_record[epoch_id-1] = self.evaluate(dev_loader)
            print("epoch ", epoch_id, "dev loss: ", dev_loss_record[epoch_id-1])

            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.output_path, 'checkpoint_{}.pth'.format(epoch_id))
            )

            with open(self.config.output_path + "loss.txt", "w") as f:
                f.write("train loss: \n")
                f.write(" ".join([str(loss) for loss in train_loss_record]) + "\n")
                f.write("dev loss: \n")
                f.write(" ".join([str(loss) for loss in dev_loss_record]))
    
    def train_one_epoch(self, train_dataloader):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

        self.model = self.model.to(self.config.device)
        self.model.train()
        
        alpha = 0.5
        epoch_loss = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for _, (id_tensor, mask_tensor, ASR_scores, cers, utt_ids) in loop:
            batch_loss = 0

            id_tensor = id_tensor.to(self.config.device)
            mask_tensor = mask_tensor.to(self.config.device)
            ASR_scores = ASR_scores.to(self.config.device)
            cers = cers.to(self.config.device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = self.model(
                    input_ids=id_tensor,
                    attention_mask=mask_tensor
                )
                LM_scores = torch.squeeze(output)

                final = False
                for index, current_id in enumerate(utt_ids):
                    if index == len(utt_ids)-1:
                        final = True
                    else:
                        next_id = utt_ids[index+1]
    
                    if current_id != next_id or final:
                        # new segment(new utt)
                        segment_start = utt_ids.index(current_id)
                        segment_end = len(utt_ids) if final else utt_ids.index(next_id)

                        final_scores = ASR_scores[segment_start: segment_end] + LM_scores[segment_start: segment_end]
                        
                        # 避免score太大，導致後免exponential 運算無法計算
                        shift_final_scores = final_scores - torch.mean(final_scores)

                        numerator = torch.exp(-1 * shift_final_scores)
                        denominator = torch.sum(numerator)
                        hypothesis_probilities = numerator / denominator
                                                
                        average_cer = torch.sum(cers[segment_start: segment_end]) / (segment_end - segment_start)

                        batch_loss += torch.dot(
                            hypothesis_probilities,
                            cers[segment_start: segment_end] - average_cer
                        )

                batch_loss.backward()
                optimizer.step()

            epoch_loss += batch_loss.item()

        epoch_loss /= len(train_dataloader)
        return epoch_loss

    def evaluate(self, dev_dataloader):
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        epoch_loss = 0
        loop = tqdm(enumerate(dev_dataloader), total=len(dev_dataloader))
        for _, (id_tensor, mask_tensor, ASR_scores, cers, utt_ids) in loop:

            id_tensor = id_tensor.to(self.config.device)
            mask_tensor = mask_tensor.to(self.config.device)
            ASR_scores = ASR_scores.to(self.config.device)
            cers = cers.to(self.config.device)
            with torch.no_grad():
                output = self.model(
                    input_ids=id_tensor,
                    attention_mask=mask_tensor
                )
                LM_scores = torch.squeeze(output)
                
                final = False
                for index, current_id in enumerate(utt_ids):
                    if index == len(utt_ids)-1:
                        final = True
                    else:
                        next_id = utt_ids[index+1]
    
                    if current_id != next_id or final:
                        # new segment(new utt)
                        segment_start = utt_ids.index(current_id)
                        segment_end = len(utt_ids) if final else utt_ids.index(next_id)

                        final_scores = ASR_scores[segment_start: segment_end] + LM_scores[segment_start: segment_end]
                        
                        # 避免score太大，導致後免exponential 運算無法計算
                        shift_final_scores = final_scores - torch.mean(final_scores)

                        numerator = torch.exp(-1 * shift_final_scores)
                        denominator = torch.sum(numerator)
                        hypothesis_probilities = numerator / denominator
                                                
                        average_cer = torch.sum(cers[segment_start: segment_end]) / (segment_end - segment_start)

                        epoch_loss += torch.dot(
                            hypothesis_probilities,
                            cers[segment_start: segment_end] - average_cer
                        )

        epoch_loss = epoch_loss.item()
        epoch_loss /= len(dev_dataloader)
        
        return epoch_loss

    class trainDataset(Dataset):
        def __init__(self, ids_list, attention_mask_list, ASR_score_list, cer_list, utt_list):
            self.ids_list = ids_list
            self.attention_mask_list = attention_mask_list
            self.ASR_score_list = ASR_score_list
            self.cer_list = cer_list
            self.utt_list = utt_list

        def __len__(self):
            return max(self.utt_list) + 1
        
        def __getitem__(self, utt_id):
            segment_start = self.utt_list.index(utt_id)
            final = False
            for index in range(segment_start, len(self.utt_list)):
                if index == len(self.utt_list)-1:
                        final = True
                else:
                    next_utt_id = self.utt_list[index+1]
                    
                if utt_id != next_utt_id or final:
                    segment_end = len(self.utt_list) if final else self.utt_list.index(next_utt_id)
                    break

            ids_list = self.ids_list[segment_start: segment_end]

            attention_mask_list = self.attention_mask_list[segment_start: segment_end]
            
            ASR_score_list = self.ASR_score_list[segment_start: segment_end]
            
            cer_list = self.cer_list[segment_start: segment_end]
            
            utt_id_list = self.utt_list[segment_start: segment_end]

            return ids_list, attention_mask_list, ASR_score_list, cer_list, utt_id_list
