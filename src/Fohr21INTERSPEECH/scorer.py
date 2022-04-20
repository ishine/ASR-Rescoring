import abc
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from models.Fohr21INTERSPEECH import BERTsem, BERTalsem
from util.dataparser import DataParser
from util.saving import json_saving

class BaseScorer(metaclass=abc.ABCMeta):
    def __init__(self, config) -> None:

        print("Setting scorer ...")

        self.seed = config.seed
        self.device = config.device
        
        self.asr_data_file = config.asr_data_file
        self.model_file = config.model_file
        self.output_file = config.output_file

        self.preprocess_config = config.preprocess
        self.model_config = config.model
        self.dataloader_config = config.dataloader

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

    def prepare_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate,
            batch_size=self.dataloader_config.batch_size,
            num_workers=self.dataloader_config.num_worker
        )
        return dataloader

    @abc.abstractmethod
    def load_model(self):
        return NotImplementedError

    @abc.abstractmethod
    def run_one_epoch(self):
        return NotImplementedError

    def score(self):

        print("Parsing inference data ...")
        self.inference_data = self.parse_data(self.asr_data_file)
        
        print("Loading tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_config.bert
        )

        print("Preparing inference dataset and dataloader ...")
        self.inference_dataset = self.prepare_dataset(self.inference_data)
        self.inference_dataloader = self.prepare_dataloader(self.inference_dataset)

        print("Loading model ...")
        self.model = self.load_model()

        print("Start scoring ...")
        self.run_one_epoch(self.inference_dataloader)
        json_data = self.inference_data.to_json()
        json_saving(self.output_file, json_data)

        print("Scoring finish.")


class BERTsemScorer(BaseScorer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_dataset(self, data):
        input_ids = []
        token_type_ids = []
        score_save_pos = []
        for utt in tqdm(data.utt_set, total=len(data.utt_set)): 

            for _ in range(len(utt.hyps)):

                hyp_i = utt.hyps.pop(0)
                for hyp_j in utt.hyps:

                    hyp_i_token_seq = self.tokenizer.tokenize(hyp_i.text)
                    hyp_j_token_seq = self.tokenizer.tokenize(hyp_j.text)

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
                    score_save_pos.append([
                        utt.index * data.n_best + hyp_i.index,
                        utt.index * data.n_best + hyp_j.index
                    ])
                    
                utt.hyps.append(hyp_i)

        attention_masks = [[1]*len(row) for row in input_ids]
        
        return self.MyDataset(input_ids, token_type_ids, score_save_pos, attention_masks)
    
    def collate(self, data):
        input_ids_tensors, token_type_ids_tensors, \
            attention_masks_tensors, score_save_pos = zip(*data)
        input_ids_tensor = pad_sequence(input_ids_tensors, batch_first=True)
        token_type_ids_tensor = pad_sequence(token_type_ids_tensors, batch_first=True)
        attention_masks_tensor = pad_sequence(attention_masks_tensors, batch_first=True)

        score_save_pos = torch.tensor(score_save_pos)
        score_save_pos = score_save_pos.transpose(1, 0)
        return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, score_save_pos
    
    def load_model(self):
        self.model = BERTsem(self.model_config)
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint)
        return self.model
    
    def run_one_epoch(self, dataloader):
        scores = torch.zeros(self.inference_data.num_utt * self.inference_data.n_best).to(self.device)

        self.model = self.model.to(self.device)
        self.model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, (input_ids, token_type_ids, attention_masks, score_save_pos) in loop:
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            score_save_pos = score_save_pos.to(self.device)
            
            with torch.set_grad_enabled(False):
                output = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                )

            scores = scores.index_add(dim=0, index=score_save_pos[0], source=output)
            scores = scores.index_add(dim=0, index=score_save_pos[1], source=1-output)

        scores = scores.reshape(self.inference_data.size())
        self.inference_data.update_scores(scores.tolist())
    
    class MyDataset(Dataset):
        def __init__(self, input_ids, token_type_ids, score_save_pos, attention_masks):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.score_save_pos = score_save_pos
            self.attention_masks = attention_masks

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            token_type_ids_tensor = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, self.score_save_pos[idx]

class BERTalsemScorer(BaseScorer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_dataset(self, data):
        input_ids = []
        token_type_ids = []
        scores = []
        score_save_pos = []
        for utt in tqdm(data.utt_set, total=len(data.utt_set)): 

            for _ in range(len(utt.hyps)):

                hyp_i = utt.hyps.pop(0)
                for hyp_j in utt.hyps:

                    hyp_i_token_seq = self.tokenizer.tokenize(hyp_i.text)
                    hyp_j_token_seq = self.tokenizer.tokenize(hyp_j.text)

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
                    scores.append([hyp_i.score, hyp_j.score])
                    score_save_pos.append([
                        utt.index * data.n_best + hyp_i.index,
                        utt.index * data.n_best + hyp_j.index
                    ])
                    
                utt.hyps.append(hyp_i)

        attention_masks = [[1]*len(row) for row in input_ids]
        
        return self.MyDataset(input_ids, token_type_ids, scores, score_save_pos, attention_masks)
    
    def collate(self, data):
        input_ids_tensors, token_type_ids_tensors, scores_tensors,\
            attention_masks_tensors, score_save_pos = zip(*data)
        input_ids_tensor = pad_sequence(input_ids_tensors, batch_first=True)
        token_type_ids_tensor = pad_sequence(token_type_ids_tensors, batch_first=True)
        attention_masks_tensor = pad_sequence(attention_masks_tensors, batch_first=True)
        scores_tensor = torch.stack(scores_tensors)
        score_save_pos = torch.tensor(score_save_pos)
        score_save_pos = score_save_pos.transpose(1, 0)
        return input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, scores_tensor, score_save_pos
    
    def load_model(self):
        self.model = BERTalsem(self.model_config)
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint)
        return self.model
    
    def run_one_epoch(self, dataloader):
        sav_scores = torch.zeros(self.inference_data.num_utt * self.inference_data.n_best).to(self.device)

        self.model = self.model.to(self.device)
        self.model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, (input_ids, token_type_ids, attention_masks, scores, score_save_pos) in loop:
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            scores = scores.to(self.device)
            score_save_pos = score_save_pos.to(self.device)
            
            with torch.set_grad_enabled(False):
                output = self.model(
                    device=self.device,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    scores=scores
                ).squeeze(dim=1)

            sav_scores = sav_scores.index_add(dim=0, index=score_save_pos[0], source=output)
            sav_scores = sav_scores.index_add(dim=0, index=score_save_pos[1], source=1-output)

        sav_scores = sav_scores.reshape(self.inference_data.size())
        self.inference_data.update_scores(sav_scores.tolist())
    
    class MyDataset(Dataset):
        def __init__(self, input_ids, token_type_ids, scores, score_save_pos, attention_masks):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.scores = scores
            self.score_save_pos = score_save_pos
            self.attention_masks = attention_masks

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            input_ids_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
            token_type_ids_tensor = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            scores_tensor = torch.tensor(self.scores[idx], dtype=torch.float)
            attention_masks_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return input_ids_tensor, token_type_ids_tensor, scores_tensor, attention_masks_tensor, self.score_save_pos[idx]

