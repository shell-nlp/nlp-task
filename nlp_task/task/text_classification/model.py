import torch
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer


class TextClassificationModel(torch.nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = output.loss
        return {
            "loss": loss,
        }
