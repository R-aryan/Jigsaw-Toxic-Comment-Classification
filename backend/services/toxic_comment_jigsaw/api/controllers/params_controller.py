from injector import inject

from backend.services.toxic_comment_jigsaw.api.controllers.controller import Controller


class ParamsController(Controller):
    @inject
    def __init__(self):
        pass

    def post(self):
        pass

    def get(self):
        pass
