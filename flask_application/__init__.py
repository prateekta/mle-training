from flask import Flask
from flask_login import LoginManager

from .config.production import Config

app = Flask(__name__)
app.config.from_object(Config)
login = LoginManager(app)
from flask_application import app  # isort:skip # noqa
