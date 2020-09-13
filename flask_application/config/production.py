import os


class Config(object):
    SECRET_KEY = (
        os.environ.get("SECRET_KEY") or "5f352379324c22463451387a0aec5d2f"
    )
    DEBUG = False
