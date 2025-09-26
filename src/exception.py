import sys
import logging
from src.logger import setup_logger

# Always configure logger early
logger = setup_logger()

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

# # Test block
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logger.error("Division by zero error")
#         raise CustomException(e, sys)