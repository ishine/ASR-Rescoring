from transformers import BertModel

bert = BertModel.from_pretrained("bert-base-chinese", attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3)
print(bert.config)