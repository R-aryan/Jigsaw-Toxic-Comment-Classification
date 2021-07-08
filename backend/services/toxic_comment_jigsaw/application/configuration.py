from injector import Module, singleton

from backend.common.logging.console_loger import ConsoleLogger
from backend.common.logging.logger import Logger
from backend.services.toxic_comment_jigsaw.application.ai.inference.prediction import Prediction
from backend.services.toxic_comment_jigsaw.application.ai.training.src.preprocess import Preprocess
from backend.services.toxic_comment_jigsaw.application.ai.settings import Settings


class Configuration(Module):
    def configure(self, binder):
        logger = ConsoleLogger(filename=Settings.LOGS_DIRECTORY)
        binder.bind(Logger, to=logger, scope=singleton)
        binder.bind(Prediction, to=Prediction(preprocess=Preprocess(), logger=logger), scope=singleton)
