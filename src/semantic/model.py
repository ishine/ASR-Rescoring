import torch
from transformers import BertModel

class BERTsem(torch.nn.Module):
    def __init__(self, config):
        super(BERTsem, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert)
        self.linear = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_last_hidden_state = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict =True
        ).last_hidden_state

        cls_embedding = bert_last_hidden_state[:, 0, :]
        linear_output = self.linear(cls_embedding)
        output = self.sigmoid(linear_output).squeeze(dim=1)
        return output


class BERTalsem(torch.nn.Module):
    def __init__(self, config):
        super(BERTalsem, self).__init__()
        self.drop_layer = torch.nn.Dropout(p=0.3)
        self.bert = BertModel.from_pretrained(config.bert)
        self.BiLSTM = torch.nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            dropout=0.3,
            bidirectional=True
        )
        self.first_FC = torch.nn.Linear(in_features=256, out_features=128)
        self.relu = torch.nn.ReLU()
        self.second_FC = torch.nn.Linear(in_features=132, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask, am_scores, lm_scores):
        bert_last_hidden_state = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict =True
        ).last_hidden_state

        return 0