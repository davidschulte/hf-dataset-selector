import logging


def get_logger(name, output_filepath):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # file_handler = logging.FileHandler(Path(output_filepath).as_posix())
    file_handler = logging.FileHandler(output_filepath)
    file_handler_formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')
    file_handler.setFormatter(file_handler_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_handler_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

