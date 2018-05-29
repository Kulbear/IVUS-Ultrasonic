import argparse
import json
import os

from bunch import Bunch


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    run = config.run
    version = config.version
    config.summary_dir = os.path.join(
        '../experiments_v{}_run_{}'.format(version, run), config.exp_name[:-2],
        'summary/')
    config.checkpoint_dir = os.path.join(
        '../experiments_v{}_run_{}'.format(version, run), config.exp_name[:-2],
        'checkpoint/')
    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print('Creating directories error: {0}'.format(err))
        exit(-1)
