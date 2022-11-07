import logging

def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
            format='%(asctime)s | %(levelname)s: %(message)s',
            level=logging.DEBUG, filemode="w")
    
