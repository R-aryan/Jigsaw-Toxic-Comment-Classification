from flask_injector import FlaskInjector
from backend.services.toxic_comment_jigsaw.api.server import server
from backend.services.toxic_comment_jigsaw.application.configuration import Configuration
from backend.services.toxic_comment_jigsaw.api.controllers.params_controller import ParamsController

api_name = '/toxic_comment/api/v1/'

server.api.add_resource(ParamsController, api_name + 'predict', methods=["GET", "POST"])

flask_injector = FlaskInjector(app=server.app, modules=[Configuration])
