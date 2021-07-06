from backend.common.logging.logger import Logger


class ConsoleLogger(Logger):
    def __init__(self):

        super(ConsoleLogger, self).__init__()

    def debug(self, message, file_id='', context=''):
        print('DEBUG: '+message)

    def info(self, message, file_id='', context=''):
        print('INFO: '+message)

    def warning(self, message, file_id='', context=''):
        print('WARNING: '+message)

    def error(self, message, file_id='', context=''):
        print('ERROR: '+message)

    def fatal(self, message, file_id='', context=''):
        print('FATAL: '+message)