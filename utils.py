import os
import logging
import json


def load_config(config, path_or_dict):
    if type(path_or_dict) == dict:
        new_config = path_or_dict
    elif os.path.isfile(path_or_dict):
        with open(path_or_dict) as f:
            new_config = json.load(f)
    else:
        raise ValueError('path_or_dict can be either a dict or path to a json file')
    fixed_configs = {key: config[key] for key in config.fixed_configs}

    config.update(new_config)
    config.update(fixed_configs)

    return config


def set_logger(config):
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.exp_dir, 'log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)


def prepare_experiment(config):
    """make sub-directories for each experiment and update the config"""

    for key, value in vars(config).items():
        if key.endswith('dir') and not os.path.exists(value):
            os.mkdir(value)

    model_dir = os.path.join(config.experiments_dir, config.model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    exp_dir = os.path.join(config.experiments_dir, config.model_name, config.experiment_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    sub_dir_names = ['summaries', 'checkpoints', 'results']
    for dir_name in sub_dir_names:
        if not os.path.exists(os.path.join(exp_dir, dir_name)):
            os.mkdir(os.path.join(exp_dir, dir_name))
    dir_dict = {
        'model_dir': model_dir,
        'exp_dir': exp_dir,
        'summary_dir': os.path.join(exp_dir, 'summaries'),
        'ckpt_dir': os.path.join(exp_dir, 'checkpoints'),
        'result_dir': os.path.join(exp_dir, 'results')
    }
    config.update(dir_dict)

    return config
