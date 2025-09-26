import logging
import os

def setup_logger(name: str = "mlproject", level: int = logging.INFO) -> logging.Logger:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{name}.log')

    # Remove existing root handlers (very important for testing/reloading)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging to file
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='a'
    )

    # Also add console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    return logging.getLogger(name)  # or just return logging.getLogger()




# # Test the logger setup
# if __name__ == "__main__":
#     test_logger = setup_logger("mlproject", logging.DEBUG)
#     test_logger.info("This is an info message.")
#     test_logger.error("This is an error message.")
#     test_logger.debug("This is a debug message.")