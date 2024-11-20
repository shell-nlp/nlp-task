import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import (
    BertPooler,
    BertForTokenClassification,
    BertForMaskedLM,
)
from transformers import AutoTokenizer


class TextCorrectionModel(torch.nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.mlm_model = BertForMaskedLM.from_pretrained(ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

        config = self.mlm_model.config
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids, labels, corr_labels):
        output = self.mlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
        )

        bert_last_hidden_states = output.hidden_states[-1]
        sequence_output = self.dropout(bert_last_hidden_states)
        corr_logits = self.classifier(sequence_output)
        masked_lm_loss = output.loss
        total_loss = masked_lm_loss

        mlm_logits = output.logits
        if corr_labels is not None:
            loss_fct = CrossEntropyLoss()
            corr_loss = loss_fct(corr_logits.view(-1, 2), corr_labels.view(-1))
            total_loss += corr_loss

        return {"loss": total_loss, "logits": mlm_logits}
