import torch
from transformers import BertTokenizer, BartModel, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
print(model.config)

inputs = tokenizer(
    ["早上好", "早上[MASK]"],
    return_tensors="pt",
    padding=True,
    truncation=True
)
print(inputs)

model.eval()
output = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    return_dict=True
).logits
print(output.size())

'''
linear_layer = torch.nn.Linear(
    in_features=model.config.d_model,
    out_features=model.config.vocab_size)
logits = linear_layer(output.last_hidden_state)
print(logits.size())
'''

predict_ids = torch.argmax(output, dim=-1)
print(predict_ids.size())
for pred in predict_ids.tolist():
    print(tokenizer.convert_ids_to_tokens(pred))