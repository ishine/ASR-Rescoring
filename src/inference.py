import sys
import json
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

from models.token_bert_lm import TokenBertLM
from models.sentence_bert_lm import SentenceBertLM

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

def load_model(model_name, model_weight_path=None):
    model = BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    if model_weight_path != None:
        checkpoint = torch.load(model_weight_path)
        model.load_state_dict(checkpoint)
    return model

def get_recog_data(data_path, type, max_utts=-1, flatten=True):

    json_data = json.load(open(data_path, "r", encoding="utf-8"))

    if type == "all":
        corpus = {}
        for utt_count, (utt_id, recog_content) in enumerate(json_data.items(), 1):
            corpus.update({utt_id: recog_content})
            if utt_count == max_utts:
                break
        return corpus

    elif type == "ref":
        corpus = []
        for utt_count, (utt_id, recog_content) in enumerate(json_data.items(), 1):
            corpus.append(recog_content["ref"])
            if utt_count == max_utts:
                break
        return corpus

    elif type == "hyp_score":
        all_scores = []
        for utt_count, (utt_id, recog_content) in enumerate(json_data.items(), 1):
            utt_scores = []
            for hyp_id, hyp_content in recog_content.items():
                if hyp_id == "ref":
                    continue
                utt_scores.append(hyp_content["score"])
            all_scores.append(utt_scores)
            if utt_count == max_utts:
                break
        return flatten_2dlist(all_scores) if flatten else all_scores

    elif type == "hyp_text":
        all_text = []
        for utt_count, (utt_id, recog_content) in enumerate(json_data.items(), 1):
            utt_text = []
            for hyp_id, hyp_content in recog_content.items():
                if hyp_id == "ref":
                    continue
                utt_text.append(hyp_content["text"])
            all_text.append(utt_text)
            if utt_count == max_utts:
                break
        return flatten_2dlist(all_text) if flatten  else all_text

    elif type == "cer":
        all_cer = []
        for utt_count, (utt_id, recog_content) in enumerate(json_data.items(), 1):
            utt_cer = []
            for hyp_id, hyp_content in recog_content.items():
                if hyp_id == "ref":
                    continue
                utt_cer.append(hyp_content["cer"])
            all_cer.append(utt_cer)
            if utt_count == max_utts:
                break
        return flatten_2dlist(all_cer) if flatten  else all_cer

def flatten_2dlist(input_list):
    output_list = []
    for l in input_list:
        output_list += l
    return output_list


def collate(data):
    ids_tensor, attention_mask, masked_token_pos, masked_token_id, seq_id = zip(*data)
    
    ids_tensor = pad_sequence(ids_tensor, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    return ids_tensor, attention_mask, masked_token_pos, masked_token_id, seq_id

class TokenlevelScoring():
    def __init__(self, config):
        self.config = config
        '''
        self.model = TokenBertLM(
            load_model(
                self.config.model,
                self.config.model_weight_path
            )
        )
        '''
        self.model = load_model(
                self.config.model,
                self.config.model_weight_path
        )
        self.tokenizer = load_tokenizer(self.config.model)
        self.recog_data = get_recog_data(self.config.asr_data_path, type="all", max_utts=self.config.max_utts)
        
        self.UttID_and_HypID_to_SeqID = {}
        seq_id = 0
        for utt_id, recog_content in self.recog_data.items():
            #recog_content.pop("ref")
            for hyp_id, _ in recog_content.items():
                if hyp_id == "ref":
                    continue
                self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)] = seq_id
                seq_id += 1
        
        self.corpus_len = seq_id

    def prepare_inference_set(self):
        ids_set = []
        attention_mask_set = []
        masked_token_pos_set = []
        masked_token_id_set = []
        seq_id_set = []

        # 把每個seq複製和它長度一樣的次數
        for utt_id, recog_content in self.recog_data.items():
            
            #recog_content.pop("ref")
            for hyp_id, hyp_content in recog_content.items():
                if hyp_id == "ref":
                    continue

                token_seq = self.tokenizer.tokenize(hyp_content["text"])
                for mask_pos in range(len(token_seq)):
                    seq_id_set.append(self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)])

                    masked_token_pos_set.append(mask_pos+1)

                    masked_token_id_set.append(self.tokenizer.convert_tokens_to_ids(token_seq[mask_pos]))
                    
                    masked_token_seq = mask_sentence(self.tokenizer, token_seq, mask_pos)
                    ids_set.append(
                        self.tokenizer.convert_tokens_to_ids(
                            ["[CLS]"] + masked_token_seq + ["[SEP]"]
                        )
                    )
                    attention_mask_set.append([1]*len(["[CLS]"] + masked_token_seq + ["[SEP]"]))
        
        self.inference_set = self.InferenceDataset(
            ids_set, attention_mask_set, masked_token_pos_set, masked_token_id_set, seq_id_set)
        
        return self.inference_set
    
    def prepare_inference_loader(self):
        self.inference_loader = DataLoader(
            dataset=self.inference_set,
            collate_fn=collate,
            batch_size=self.config.batch_size,
        )
        return self.inference_loader
    
    def score(self):
        self.scores = np.array([0.]*self.corpus_len)
        self.model.to(self.config.device)
        self.model.eval()

        loop = tqdm(enumerate(self.inference_loader), total=len(self.inference_loader))

        for _, (ids_tensor, attention_mask, masked_token_pos, masked_token_id, seq_id) in loop:      
            ids_tensor = ids_tensor.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)

            # 每個seq在這個batch中的位置，index從0開始，所以第一個seq的位置=0
            seq_pos_in_batch = list(range(len(ids_tensor)))

            with torch.no_grad():
                # output_logits -> size: [batch size(seq num), seq length(token num), vocab size]
                output_logits = self.model(
                    input_ids = ids_tensor,
                    attention_mask = attention_mask
                )[0]

                # 取出這個batch中每個seq個被mask的token的位置對應的output logits
                output_logits = output_logits[seq_pos_in_batch, masked_token_pos,:]

                # 對最後一維，也就是長度是vocab size的那維做softmax，再把全部數值取log
                output_score = output_logits.log_softmax(dim=-1)

                # 利用masked_token_id找出 masked token在vocab size那維的位置
                output_score = output_score[seq_pos_in_batch, masked_token_id]

                # seq_id 代表 batch中每個seq屬於「哪個utt中的哪個hypothesis sentence的獨一無二的id」
                np.add.at(self.scores, list(seq_id), output_score.cpu().numpy())

        # 寫入存檔
        output_json = self.recog_data
        for utt_id, recog_content in self.recog_data.items():
            #recog_content.pop("ref")
            for hyp_id, _ in recog_content.items():
                if hyp_id == "ref":
                    continue
                seq_id = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                output_json[utt_id][hyp_id]["score"] = self.scores[seq_id]
    
        
        with open(self.config.output_path, "w", encoding="utf8") as f:
            output_json = json.dump(output_json, f, ensure_ascii=False, indent=4)

    class InferenceDataset(Dataset):
        def __init__(self, ids_set, attention_mask_set, masked_token_pos_set, masked_token_id_set, seq_id_set):
            self.ids_set = ids_set
            self.attention_mask_set = attention_mask_set
            self.masked_token_pos_set = masked_token_pos_set
            self.masked_token_id_set = masked_token_id_set
            self.seq_id_set = seq_id_set
        
        def __len__(self):
            return len(self.ids_set)
        
        def __getitem__(self, idx):
            ids_tensor = torch.tensor(self.ids_set[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.attention_mask_set[idx], dtype=torch.long)
        
            return ids_tensor, attention_mask, self.masked_token_pos_set[idx],\
                self.masked_token_id_set[idx], self.seq_id_set[idx]

class SentencelevelScoring():
    def __init__(self, config):
        self.config = config
        self.model = SentenceBertLM(
            bert=BertModel.from_pretrained(config.model)
        )
        self.tokenizer = load_tokenizer(self.config.model)
        self.recog_data = get_recog_data(self.config.asr_data_path, type="all", max_utts=self.config.max_utts)
        
        self.UttID_and_HypID_to_SeqID = {}
        seq_id = 0
        for utt_id, recog_content in self.recog_data.items():
            for hyp_id, _ in recog_content.items():
                if hyp_id == "ref":
                    continue
                self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)] = seq_id
                seq_id += 1
        
        self.corpus_len = seq_id

    def prepare_inference_set(self):
        ids_set = []
        attention_mask_set = []
        seq_id_set = []

        # 把每個seq複製和它長度一樣的次數
        for utt_id, recog_content in self.recog_data.items():
            
            #recog_content.pop("ref")
            for hyp_id, hyp_content in recog_content.items():
                if hyp_id == "ref":
                    continue

                token_seq = self.tokenizer.tokenize(hyp_content["text"])

                seq_id_set.append(self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)])
                
                ids_set.append(
                    self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + token_seq + ["[SEP]"]
                    )
                )

                attention_mask_set.append([1]*len(["[CLS]"] + token_seq + ["[SEP]"]))
        
        self.inference_set = self.InferenceDataset(
            ids_set, attention_mask_set, seq_id_set)
        
        return self.inference_set
    
    def prepare_inference_loader(self):
        self.inference_loader = DataLoader(
            dataset=self.inference_set,
            collate_fn=self.collate,
            batch_size=self.config.batch_size,
        )
        return self.inference_loader
    
    def score(self):
        self.scores = np.array([0.]*self.corpus_len)
        self.model.to(self.config.device)
        self.model.eval()

        loop = tqdm(enumerate(self.inference_loader), total=len(self.inference_loader))

        for _, (ids_tensor, attention_mask, seq_id) in loop:      
            ids_tensor = ids_tensor.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)

            # 每個seq在這個batch中的位置，index從0開始，所以第一個seq的位置=0
            seq_pos_in_batch = list(range(len(ids_tensor)))

            with torch.no_grad():
                # output_score -> size: [batch size(seq num), score]
                output_score = self.model(
                    input_ids = ids_tensor,
                    attention_mask = attention_mask
                )
                output_score = torch.squeeze(output_score)
                np.add.at(self.scores, list(seq_id), output_score.cpu().numpy())

        # 寫入存檔
        output_json = self.recog_data
        for utt_id, recog_content in self.recog_data.items():
            #recog_content.pop("ref")
            for hyp_id, _ in recog_content.items():
                if hyp_id == "ref":
                    continue
                seq_id = self.UttID_and_HypID_to_SeqID[(utt_id, hyp_id)]
                output_json[utt_id][hyp_id]["score"] = self.scores[seq_id]
    
        
        with open(self.config.output_path, "w", encoding="utf8") as f:
            output_json = json.dump(output_json, f, ensure_ascii=False, indent=4)
    
    def collate(self, data):
        ids_tensor, attention_mask, seq_id = zip(*data)
    
        ids_tensor = pad_sequence(ids_tensor, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        return ids_tensor, attention_mask, seq_id

    class InferenceDataset(Dataset):
        def __init__(self, ids_set, attention_mask_set, seq_id_set):
            self.ids_set = ids_set
            self.attention_mask_set = attention_mask_set
            self.seq_id_set = seq_id_set
        
        def __len__(self):
            return len(self.ids_set)
        
        def __getitem__(self, idx):
            ids_tensor = torch.tensor(self.ids_set[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.attention_mask_set[idx], dtype=torch.long)
        
            return ids_tensor, attention_mask, self.seq_id_set[idx]