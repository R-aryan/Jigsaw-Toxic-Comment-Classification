import torch
from transformers import BertTokenizer


class BERTDataset:
    def __init__(self, texts, targets, max_len=128):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        pass

    def preprocessing_for_bert(self,data):
        input_ids = []
        attention_masks = []
