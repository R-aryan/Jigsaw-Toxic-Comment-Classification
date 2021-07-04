import os

import torch


class Settings:
    PROJ_NAME = 'Jigsaw-Toxic-Comment-Classification'
    root_path = os.getcwd().split(PROJ_NAME)[0] + "\\" + PROJ_NAME + "\\"
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    EPOCHS = 10

    # training data directory
    TRAIN_PATH = root_path + "\\training\\data\\train.csv"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768
    hidden_dim = 64
    output_dim = 6
    bert_model_name = 'bert-base-uncased'

