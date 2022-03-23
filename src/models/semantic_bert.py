import torch
from transformers import BertModel

class SemanticBert(torch.nn.Module):
    def __init__(self, bert_model):
        super(SemanticBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
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
        output = self.sigmoid(linear_output)
        return output