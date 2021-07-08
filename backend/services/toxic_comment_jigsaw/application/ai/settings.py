import os

import torch


class Settings:
    PROJ_NAME = 'Jigsaw-Toxic-Comment-Classification'
    root_path = os.getcwd().split(PROJ_NAME)[0] + PROJ_NAME + "\\"
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 2
    RANDOM_STATE = 42
    MODEL_PATH = root_path + 'model.bin'
    TRAIN_NUM_WORKERS = 4
    VAL_NUM_WORKERS = 1

    APPLICATION_PATH = root_path + "backend\\services\\toxic_comment_jigsaw\\application\\"

    # training data directory
    TRAIN_DATA = APPLICATION_PATH + "ai\\training\\data\\train.csv"

    # test data directory
    TEST_DATA = APPLICATION_PATH + "ai\\training\\data\\test.csv"

    # weights path
    WEIGHTS_PATH = APPLICATION_PATH + "ai\\weights\\bert_base_uncased\\toxic_model.bin"

    # setting up logs path
    LOGS_DIRECTORY = root_path + "backend\\services\\toxic_comment_jigsaw\\logs\\logs.txt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768
    hidden_dim = 50
    output_dim = 6
    bert_model_name = 'bert-base-uncased'

    # mapping of columns
    column_label = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
