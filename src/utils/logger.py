import os
import sys
import loguru


def get_logger(outputfile):
    logger = loguru.logger
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile is not None:
        if not os.path.exists(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger