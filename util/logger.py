import logging
import config


def setup_logger(logger_name, level=logging.DEBUG):
    """Set up a logger that outputs to both console and file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler and set level to DEBUG
    file_handler = logging.FileHandler(logger_name)
    file_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


printf = setup_logger(config.path_name + "log.txt")
