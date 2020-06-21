import logging
import os


def create_logger(logging_path, name):
    os.makedirs(logging_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "'%(asctime)s %(name)-12s %(levelname)-8s %(message)s'"
    )
    file_handler = logging.FileHandler(os.path.join(logging_path, name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
