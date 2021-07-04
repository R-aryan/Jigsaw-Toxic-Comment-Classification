import torch
from transformers import BertTokenizer

from training.settings import Settings


class BERTDataset:
    def __init__(self, texts, targets):
        self.settings = Settings
        self.texts = texts
        self.targets = targets
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sent = str(self.texts[item])
        inputs = self.tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=self.settings.MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'targets': torch.tensor(self.targets[item])
        }
