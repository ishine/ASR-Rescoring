import torch
from torch.nn.functional import one_hot
from transformers import BertModel, BertTokenizer

class NbestAlignBert(torch.nn.Module):
    def __init__(self, bert):
        super(NbestAlignBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.linear = torch.nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=self.bert.config.vocab_size
        )

    def forward(self, input_ids, attention_mask, token_type_ids, cls_pos, ref_ids):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        ).last_hidden_state
        bert_cls_output = bert_output[range(len(bert_output)), cls_pos, :]

        logits = self.linear(bert_cls_output)
        logits = torch.softmax(logits)

        ref_logits = one_hot(ref_ids, self.bert.config.vocab_size)
        print(logits.size(), ref_logits.size())
        print(logits, ref_logits)

        return logits