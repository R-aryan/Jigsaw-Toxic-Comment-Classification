import torch
import pandas as pd
import numpy as np
from injector import inject
from transformers import BertTokenizer

from backend.services.toxic_comment_jigsaw.application.ai.training.src.preprocess import Preprocess
from backend.services.toxic_comment_jigsaw.application.ai.settings import Settings
from backend.services.toxic_comment_jigsaw.application.ai.model import BERTClassifier


class Prediction:
    @inject
    def __init__(self, preprocess: Preprocess):
        self.settings = Settings
        self.preprocess = preprocess

        self.__model = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

        self.load_model_weights()

    def load_model_weights(self):
        try:
            print("-------Loading Bert Base Model------")
            self.__model = BERTClassifier()
            print("-------Bert Base Model Successfully Loaded---- \n\n")

            print('Loading Model Weights----!!')
            self.__model.load_state_dict(torch.load(self.settings.WEIGHTS_PATH,
                                                    map_location=torch.device(self.settings.DEVICE)))
            self.__model.to(self.settings.DEVICE)
            self.__model.eval()
            print('Successfully Model Weights----!!')

        except BaseException as ex:
            print("Following Exception Occurred---!! ", str(ex))

    def preprocessing_for_bert(self, data):
        data = self.preprocess.clean_text(data)
        encoded_text = self.tokenizer.encode_plus(
            text=data,
            add_special_tokens=True,
            max_length=self.settings.MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        # converting to tensors and moving tensors to device
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)

        input_ids = input_ids.to(self.settings.DEVICE)
        attention_mask = attention_mask.to(self.settings.DEVICE)

        return input_ids, attention_mask

    def __predict(self, data):
        output = None
        try:
            input_ids, attention_mask = tuple(self.preprocessing_for_bert(data))
            with torch.no_grad():
                logits = self.__model(input_ids=input_ids,
                                      token_type_ids=None,
                                      attention_mask=attention_mask)

            output = logits.sigmoid().cpu().detach().numpy()

        except BaseException as ex:
            print("Following Exception Occurred---!! ", str(ex))

        return output

    def run_inference(self, data):
        print("Running Inference---!! \n")
        result = self.__predict(data)

        output = pd.DataFrame(result, columns=self.settings.column_label)

        return data, output
