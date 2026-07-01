import sys
import logging
import os
from datetime import datetime

# Create logs directory
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ],
    force=True
)

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = str(error_message)
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (f"Error occurred in script: [{self.file_name}] "
                f"at line number: [{self.lineno}] "
                f"error message: [{self.error_message}]")

if __name__ == '__main__':
    try:
        logging.info("Enter the try block")
        a = 1 / 0
        print('This will not be printed', a)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        raise NetworkSecurityException(e, sys)
