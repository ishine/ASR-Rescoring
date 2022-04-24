import torch
from transformers import BertModel

class RescoreBert(torch.nn.Module):
    def __init__(self, bert):
        super(RescoreBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.linear = torch.nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=1
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        cls = bert_output.last_hidden_state[:, 0, :]
        lm_score = self.linear(cls).squeeze(dim=-1)
        return lm_score