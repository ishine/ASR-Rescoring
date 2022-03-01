import torch

class SentenceBertLM(torch.nn.Module):
    def __init__(self, bert):
        super(SentenceBertLM, self).__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # bert_output[0] 就是 bert_output.logits
        cls = bert_output[0][:, 0, :]
        output = self.linear(cls)
        return output