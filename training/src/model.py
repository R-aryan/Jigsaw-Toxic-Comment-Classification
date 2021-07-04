import torch
import torch.nn as nn
from transformers import BertModel

from settings import Settings


class BERTClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        self.settings = Settings
        self.input_dim = self.settings.input_dim
        self.hidden_dim = self.settings.hidden_dim
        self.output_dim = self.settings.output_dim

        # loading the bert model
        self.bert = BertModel.from_pretrained(self.settings.bert_model_name)

        # adding custom layers according to the problem statement
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits
