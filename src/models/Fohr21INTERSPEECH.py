import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class BERTsem(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=config.bert
        )
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
        super().__init__()
        self.bert = BertModel.from_pretrained(
            config.bert,
            attention_probs_dropout_prob=config.dropout,
            hidden_dropout_prob=config.dropout
        )
        self.BiLSTM = torch.nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            dropout=config.dropout,
            bidirectional=True
        )
        self.first_FC = torch.nn.Linear(
            in_features=4*config.lstm_hidden_size,
            out_features=2*config.lstm_hidden_size
        )
        self.relu = torch.nn.ReLU()
        self.second_FC = torch.nn.Linear(in_features=130, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, device, input_ids, token_type_ids, attention_mask, scores):
        bert_last_hidden_state = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict =True
        ).last_hidden_state

        # 根據 attention mask 計算一個 batch 中每個 sequence 的長度 
        seq_len = [len(seq) for seq in attention_mask]

        bilstm_input = pack_padded_sequence(
            bert_last_hidden_state,
            seq_len,
            batch_first=True,
            enforce_sorted=False
        )

        bilstm_output, _ = self.BiLSTM(bilstm_input)

        bilstm_output_for_avg_pool, _ = pad_packed_sequence(
            bilstm_output,
            batch_first=True,
            padding_value=0
        )

        bilstm_output_for_max_pool, _ = pad_packed_sequence(
            bilstm_output,
            batch_first=True,
            padding_value=float("-inf")
        )

        avg_pooling = torch.sum(bilstm_output_for_avg_pool, dim=1)
        seq_len = torch.tensor(seq_len).unsqueeze(dim=1).to(device)
        avg_pooling /= seq_len

        max_pooling, _ = torch.max(bilstm_output_for_max_pool, dim=1)

        avg_max_concat = torch.cat((max_pooling, avg_pooling), dim=1)

        first_FC_output = self.first_FC(avg_max_concat)
        relu_output = self.relu(first_FC_output)

        relu_output_concat_score = torch.cat((relu_output, scores), dim=1)

        second_FC_output = self.second_FC(relu_output_concat_score)
        final_output = self.sigmoid(second_FC_output)
        return final_output