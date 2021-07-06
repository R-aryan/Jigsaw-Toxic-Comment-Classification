import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from backend.services.toxic_comment_jigsaw.application.ai.training.src.model import BERTClassifier
from backend.services.toxic_comment_jigsaw.application.ai.training.src.dataset import BERTDataset
from backend.services.toxic_comment_jigsaw.application.ai.training.src.preprocess import Preprocess
from backend.services.toxic_comment_jigsaw.application.ai.training.src.engine import Engine
from backend.services.toxic_comment_jigsaw.application.ai.training.settings import Settings

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Train:
    def __init__(self):
        # initialize required class
        self.settings = Settings
        self.engine = Engine()
        self.preprocess = Preprocess()

        # initialize required variables
        self.bert_classifier = None
        self.optimizer = None
        self.scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.total_steps = None
        self.best_accuracy = 0

    def __initialize(self):
        # Instantiate Bert Classifier
        self.bert_classifier = BERTClassifier(freeze_bert=False)
        self.bert_classifier.to(self.settings.DEVICE)

        # Create the optimizer
        self.optimizer = AdamW(self.bert_classifier.parameters(),
                               lr=5e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )
        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.total_steps)

    def crete_data_loaders(self, dataset):
        pass

    def load_data(self):
        train_df = pd.read_csv(self.settings.TRAIN_DATA).fillna("none")
        train_df['comment_text'] = train_df['comment_text'].apply(lambda x: self.preprocess.clean_text(x))
        X = list(train_df['comment_text'])
        y = np.array(train_df.loc[:, 'toxic':])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=self.settings.RANDOM_STATE)

        # training dataset
        train_dataset = BERTDataset(X_train, y_train)

        # validation dataset
        val_dataset = BERTDataset(X_val, y_val)

        self.train_data_loader = DataLoader(train_dataset,
                                            batch_size=self.settings.TRAIN_BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=self.settings.TRAIN_NUM_WORKERS)

        self.val_data_loader = DataLoader(val_dataset,
                                          batch_size=self.settings.VALID_BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=self.settings.VAL_NUM_WORKERS)

        self.total_steps = int(len(X_train) / self.settings.TRAIN_BATCH_SIZE * self.settings.EPOCHS)

    def train(self):
        for epochs in range(self.settings.EPOCHS):

            # calling the training function in engine.py file
            self.engine.train_fn(data_loader=self.train_data_loader,
                                 model=self.bert_classifier,
                                 optimizer=self.optimizer,
                                 device=self.settings.DEVICE,
                                 schedular=self.scheduler)

            # calling the evaluation function from the engine.py file to compute evaluation
            val_loss, val_accuracy = self.engine.eval_fn(data_loader=self.val_data_loader,
                                                         model=self.bert_classifier,
                                                         device=self.settings.DEVICE)

            # updating the accuracy
            if val_accuracy > self.best_accuracy:
                torch.save(self.bert_classifier.state_dict(), self.settings.MODEL_PATH)
                self.best_accuracy = val_accuracy

    def run(self):
        try:
            print("Loading and Preparing the Dataset-----!! ")
            self.load_data()
            print("Dataset Successfully Loaded and Prepared-----!! ")
            print()
            print("-" * 70)
            print("Loading and Initializing the Bert Model -----!! ")
            self.__initialize()
            print("Model Successfully Loaded and Initialized-----!! ")
            print()
            print("-" * 70)
            print("------------------Starting Training-----------!!")
            self.engine.set_seed()
            self.train()
            print("Training complete-----!!!")

        except BaseException as ex:
            print("Following Exception Occurred---!! ", str(ex))

