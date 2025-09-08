import logging
import sys
import os




def init_logging(model, level=logging.INFO):
    log_dir = './log'
    LOGGER = model
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    LOG = logging.getLogger(LOGGER)

    LOG.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{model}.log'))

    ff = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    sf = logging.Formatter('%(name)s - %(message)s')

    file_handler.setFormatter(ff)
    stream_handler.setFormatter(sf)
    
    LOG.addHandler(stream_handler)
    LOG.addHandler(file_handler)

    LOG.debug('Logger initialized')

    return LOG