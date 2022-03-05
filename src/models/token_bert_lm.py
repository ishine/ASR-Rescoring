import torch

class TokenBertLM(torch.nn.Module):
    def __init__(self, bert):
        super(TokenBertLM, self).__init__()
        self.bert = bert
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        output = self.softmax(bert_output.logits)
        return output