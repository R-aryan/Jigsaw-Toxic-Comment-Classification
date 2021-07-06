from injector import Module, singleton

from backend.common.logging.console_loger import ConsoleLogger
from backend.common.logging.logger import Logger
from backend.services.toxic_comment_jigsaw.application.ai.inference.prediction import Prediction
from backend.services.toxic_comment_jigsaw.application.ai.training.src.preprocess import Preprocess


class Configuration(Module):
    def configure(self, binder):
        logger = ConsoleLogger()
        binder.bind(Logger, to=logger, scope=singleton)
        binder.bind(Prediction, to=Prediction(preprocess=Preprocess()))
