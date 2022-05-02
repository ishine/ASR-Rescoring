import torch
from transformers import BartForConditionalGeneration, BertTokenizer


class CorrectBart(torch.nn.Module):
    def __init__(self, bart):
        super(CorrectBart, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart)
        self.tokenizer = BertTokenizer.from_pretrained(bart)

    def forward(self, input_ids, attention_mask, labels=None):
        bart_output = self.bart(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        if labels != None:
            return bart_output.loss
        else:
            predict_ids = torch.argmax(bart_output.logits, dim=-1)
            
            predictions = []
            for pred in predict_ids.tolist():
                predictions.append(self.tokenizer.convert_ids_to_tokens(pred))

            return predictions

    def generate(self, input_ids, attention_mask):
        bart_output = self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50, 
            return_dict=True
        )
        print(bart_output)
        predictions = []
        for pred in bart_output.tolist():
            pred = self.tokenizer.convert_ids_to_tokens(pred)
            pred = [token for token in pred if token not in ["[CLS]", "[SEP]", "[PAD]"]]
            pred_sentence = ""
            for token in pred:
                pred_sentence += token
            predictions.append(pred_sentence)

        return predictions