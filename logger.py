import logging
from pathlib import Path


class Logger:
    def __init__(self, log_file="./log/output.log", name="my_logger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 确保日志文件所在的目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message):
        self.logger.info(message)
