from loguru import logger

def setup_logging(log_file="app.log", level="INFO"):
    logger.remove()
    logger.add(
        log_file,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>\n"
    )

setup_logging()
__all__ = ["logger"]