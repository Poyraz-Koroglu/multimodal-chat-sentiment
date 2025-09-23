import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

class ProjectLogger:
    def __init__(self, name: str, log_dir: str = "logs", level=logging.INFO, backup_days: int = 7):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers if logger already created
        if not self.logger.handlers:
            os.makedirs(log_dir, exist_ok=True)

            # ---- File Handler (rotating daily) ----
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = TimedRotatingFileHandler(
                log_file, when="midnight", interval=1, backupCount=backup_days, encoding="utf-8"
            )
            file_handler.setLevel(level)

            # ---- Console Handler (stdout, for ELK/Docker) ----
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # ---- Log Format ----
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                "%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # ---- Add Handlers ----
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
