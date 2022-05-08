import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BertTokenizer


class CorrectBart(nn.Module):
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

        return bart_output.loss

    def generate(self, input_ids, attention_mask):
        bart_output = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        print(bart_output.logits.size())


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


class NBestAlignCorrectBart(nn.Module):
    def __init__(self, bart, alignment_embedding, n_best):
        super(NBestAlignCorrectBart, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart)
        self.tokenizer = BertTokenizer.from_pretrained(bart)
        self.embedding = nn.Embedding(
            num_embeddings=self.bart.config.vocab_size,
            embedding_dim=alignment_embedding,
            padding_idx=self.tokenizer.convert_tokens_to_ids("[PAD]")
        )
        self.linear = nn.Linear(
            in_features=n_best*alignment_embedding,
            out_features=self.bart.config.d_model
        )

    def forward(self, input_ids, attention_mask, labels=None):
        align_embed = self.embedding(input_ids)
        align_embed = align_embed.reshape(
            (
                align_embed.size(dim=0), 
                align_embed.size(dim=1),
                align_embed.size(dim=2)*align_embed.size(dim=3)
            )
        )
        bart_embedding = self.linear(align_embed)
        bart_output = self.bart(
            inputs_embeds=bart_embedding, 
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return bart_output.loss


    def generate(self, input_ids, attention_mask):
        align_embed = self.embedding(input_ids)
        align_embed = align_embed.reshape(
            (
                align_embed.size(dim=0), 
                align_embed.size(dim=1),
                align_embed.size(dim=2)*align_embed.size(dim=3)
            )
        )
        bart_embedding = self.linear(align_embed)
        bart_output = self.bart.generate(
            inputs_embeds=bart_embedding, 
            attention_mask=attention_mask,
            max_length=50, 
            return_dict=True
        ) 
        predictions = []
        for pred in bart_output.tolist():
            pred = self.tokenizer.convert_ids_to_tokens(pred)
            pred = [token for token in pred if token not in ["[CLS]", "[SEP]", "[PAD]"]]
            pred_sentence = ""
            for token in pred:
                pred_sentence += token
            predictions.append(pred_sentence)

        return predictions