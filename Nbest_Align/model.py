import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer


class NbestAlignBert(torch.nn.Module):
    def __init__(self, bert, n_best, config):
        super(NbestAlignBert, self).__init__()
        self.config = config
        self.n_best = n_best
        self.bert = BertModel.from_pretrained(bert)
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        self.linear = torch.nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=n_best
        )
        self.cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, prediction_pos, labels):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        ).last_hidden_state
        linear_output = self.linear(bert_output)
        logits = torch.softmax(linear_output, dim=-1)

        loss = 0
        pred_sentences = []
        for idx, pos in enumerate(prediction_pos):
            pred_distribution = logits[idx, pos.to(self.config.device)]
            label_distribution = torch.tensor(
                one_hot(labels[idx].to(self.config.device), self.n_best),
                dtype=torch.float
            )
            loss += self.cross_entropy_loss_fn(pred_distribution, label_distribution)
            
            pred_pos = torch.argmax(pred_distribution, dim=1)
            token_ids = []
            for p, class_num in zip(pos, pred_pos):
                token_ids.append(input_ids[idx, p + 1 + class_num])
            pred_sentences.append(self.tokenizer.convert_ids_to_tokens(token_ids))
        return {"loss": loss, "prediction": pred_sentences}