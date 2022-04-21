import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM

# load preprocessed data
input_paths = {
    "train": "preprocessed_data/train.json",
    "dev": "preprocessed_data/dev.json",
    "test": "preprocessed_data/test.json"
}

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

# training(MLM fine-tune bert) or scoring