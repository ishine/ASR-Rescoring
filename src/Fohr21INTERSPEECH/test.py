import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

input_seq1 = torch.tensor(
    [[ 1,  2,  3, 3],
     [ 4,  5,  6, 6]],
    dtype=torch.float)

input_seq2 = torch.tensor(
    [[11, 12, 13, 13],
     [14, 15, 16, 16],
     [17, 18, 19, 19]],
    dtype=torch.float)

inputs = [input_seq1, input_seq2]

batch = pad_sequence(inputs, batch_first=True)

seq_len = [len(input) for input in inputs]

packed_masked_input = pack_padded_sequence(batch, seq_len, batch_first=True, enforce_sorted=False)
print("packed_masked_input: \n", packed_masked_input)

BiLSTM = torch.nn.LSTM(
    input_size=4,
    hidden_size=2,
    num_layers=1,
    bidirectional=True
)

bilstm_output, _ = BiLSTM(packed_masked_input)
bilstm_output_for_avg_pool, _ = pad_packed_sequence(bilstm_output, batch_first=True, padding_value=0)
bilstm_output_for_max_pool, _ = pad_packed_sequence(bilstm_output, batch_first=True, padding_value=float("-inf"))
print(bilstm_output_for_avg_pool, bilstm_output_for_max_pool)

max_pooling, _ = torch.max(bilstm_output_for_max_pool, dim=1)
print(max_pooling)

avg_pooling = torch.sum(bilstm_output_for_avg_pool, dim=1) / torch.tensor(seq_len).unsqueeze(dim=1)
print(avg_pooling)

concat = torch.cat((max_pooling, avg_pooling), dim=1)
print(concat)

linear = torch.nn.Linear(8, 4)
linear_output = linear(concat)
print(linear_output)

am_socores = torch.tensor([[10, 20], [100, 200]])
lm_scores = torch.tensor([[15, 30], [150, 300]])
linear_output_cat_am_socores = torch.cat((linear_output, am_socores), dim=1)
linear_output_cat_am_socores_cat_lm_scores = torch.cat((linear_output_cat_am_socores, lm_scores), dim=1)
print(linear_output_cat_am_socores_cat_lm_scores)

linear2 = torch.nn.Linear(8, 1)
final_output = linear2(linear_output_cat_am_socores_cat_lm_scores)
print(final_output)