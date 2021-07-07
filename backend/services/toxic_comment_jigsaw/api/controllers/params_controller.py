from flask import request
from injector import inject

from backend.services.toxic_comment_jigsaw.api.controllers.controller import Controller
from backend.services.toxic_comment_jigsaw.application.ai.inference.prediction import Prediction


class ParamsController(Controller):
    @inject
    def __init__(self, prediction: Prediction):
        self.predict = prediction

    def post(self):
        try:
            req_json = request.get_json()
            response = self.prediction.run_prediction(req_json['data'])
            result = {'response': response}
            self.predict.logger.info('Request processed successfully--!!')
            return self.response_ok(result)
        except BaseException as ex:
            self.predict.logger.error('Error Occurred-- ' + str(ex))
            return self.response_error(str(ex))

    def get(self):
        return {'response': 'This is an API endpoint for toxic comment classification---!!'}
